# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ast import Tuple
from typing import TYPE_CHECKING, Optional
from tqdm import tqdm

import deepspeed
from deepspeed.utils import safe_get_full_grad
import contextlib
import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from ...data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from torch.utils.data.distributed import DistributedSampler
from ...data.converter import AlpacaDatasetConverter
from ...data.parser import DatasetAttr
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer, OrthogonalProjectionTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

# class ZeroOutGradientsCallback(TrainerCallback):
#     def on_pre_optimizer_step(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         out = safe_get_full_grad(kwargs["model"].get_output_embeddings().weight)
#         return super().on_pre_optimizer_step(args, state, control, **kwargs)

class BaseGradUpdateCallback(TrainerCallback):
    """
    周期性地重新计算 base_grads，并调用 Trainer 的 update_base_grads 接口。

    这里直接复用上面的 compute_base_gradients，确保 ZeRO-3 等场景下的维度处理逻辑一致。
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        tokenizer,
    ) -> None:
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.tokenizer = tokenizer
        # 由外部在构造 Trainer 后注入
        self.trainer = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # 每步打印当前 global_step
        local_rank = getattr(args, "local_rank", 0)
        if local_rank in (0, -1):
            print(f"[Train] global_step = {state.global_step}")
        # 未设置 base_num_steps 或者非正数，则不做任何事
        base_num_steps = getattr(args, "base_num_steps", None)
        if base_num_steps is None or base_num_steps <= 0:
            return control

        # 这里只在整除 base_num_steps 的 step 上做一次完整的锚点梯度重算
        if state.global_step == 0 or state.global_step % base_num_steps != 0:
            return control

        trainer = self.trainer
        if trainer is None:
            return control

        ds_engine = model
        if not hasattr(ds_engine, "backward"):
            ds_engine = getattr(self.trainer, "model_wrapped", model)

        # 重新计算 base_grads（使用和初始阶段同样的数据与逻辑）
        # 传入 trainer，梯度会在 hook 中自动收集并更新 base_grads
        compute_base_gradients(
            model=ds_engine,
            tokenizer=self.tokenizer,
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=self.training_args,
            finetuning_args=self.finetuning_args,
            trainer=trainer,  # 传入 trainer，梯度会在 hook 中自动收集并更新
        )

        # base_grads 已经在 finalize_base_grads_collection 中更新
        if args.should_log:
            logger.info_rank0(
                f">>> Step {state.global_step}: base_grads updated via hooks. "
                f"有效参数数目: {len(trainer.base_grads) if trainer.base_grads else 0}"
            )

        return control

def compute_base_gradients(model, tokenizer, model_args, data_args, training_args, finetuning_args, trainer=None):
    """
    针对 ZeRO-2 优化的锚点梯度计算。
    梯度通过 hook 自动收集并更新到 trainer.base_grads 中。
    
    :param trainer: 必须提供 trainer 实例，用于在 hook 中收集梯度
    """
    if trainer is None:
        raise ValueError("trainer must be provided for hook-based gradient collection")
    
    if training_args.should_log:
        print(">>> [ZeRO-2] 启动锚点梯度计算（使用 hook 收集）...")
    
    # 设置标志位跳过梯度矫正，并启动梯度收集
    if not hasattr(trainer, "_skip_gradient_projection"):
        raise ValueError("trainer must have _skip_gradient_projection attribute")
    
    trainer._skip_gradient_projection = True
    
    # 启动梯度收集
    if not hasattr(trainer, "start_base_grads_collection"):
        raise ValueError("trainer must have start_base_grads_collection method")
    
    trainer.start_base_grads_collection()
    print(">>> [BaseGrads] Using hook-based gradient collection")
    
    try:
        # 1. 准备数据 (保持原样)
        original_dataset = data_args.dataset
        data_args.dataset = [training_args.base_dataset]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", tokenizer=tokenizer)
        
        # 获取实际的 PyTorch 模型（如果是 DeepSpeed Engine，则获取 model.module）
        actual_model = getattr(model, "module", model)
        # 检查是否是 DeepSpeed Engine（有 backward 方法）
        ds_engine = model if hasattr(model, "backward") and hasattr(model, "module") else None
        
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=template,
            model=actual_model,  # data_collator 使用 actual_model
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX,
            **{"tokenizer": tokenizer}
        )

        train_dataset = dataset_module.get("train_dataset")
        num_samples = min(training_args.base_num_samples, len(train_dataset))

        train_sampler = DistributedSampler(train_dataset, shuffle=True)

        dataloader = DataLoader(
            train_dataset.select(range(num_samples)), 
            batch_size=training_args.per_device_train_batch_size, 
            collate_fn=data_collator,
            sampler = train_sampler,
            shuffle=False
        )

        # 2. 计算梯度
        actual_model.train()
        
        # 【关键修复 1】在 ZeRO-2 中计算额外梯度前，必须清空引擎状态
        if ds_engine is not None:
            actual_model.zero_grad(set_to_none=True)
            # 强制将所有梯度的共享状态设为 None，避免 Hook 误触发
            for p in actual_model.parameters():
                p.grad = None
        else:
            actual_model.zero_grad(set_to_none=True)
        
        device = training_args.device
        num_batches = len(dataloader)


        for batch in tqdm(dataloader, desc="Computing anchor gradients"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # 【关键修复 2】在 ZeRO-2 环境下，必须手动管理 grad_accumulation_steps 的假象
            # 否则 Engine 可能会在 backward 时尝试执行它不该执行的 reduce 逻辑
            # 注意：forward 使用 model（而不是 actual_model）的原因：
            # - 如果是 DeepSpeed Engine，model(**batch) 会调用 model.module(**batch) 并处理 ZeRO 分片
            # - 如果是普通模型，model 和 actual_model 是同一个对象
            outputs = model(**batch)
            loss = outputs.loss / num_batches

            if ds_engine is not None:
                # DeepSpeed Engine：必须使用 model.backward()，不能使用 loss.backward()
                # 因为 Engine 需要处理 ZeRO 分片和梯度聚合
                model.backward(loss)
            else:
                # 普通 PyTorch 模型：使用标准的 loss.backward()
                # 这会自动反向传播到 actual_model 的所有参数
                loss.backward()
            
            # 【关键修复 4】每个 batch 后立即清理 param.grad，避免梯度累积导致 OOM
            # 注意：hook 中会累加每个 batch 的梯度，所以清理 param.grad 不会影响最终结果
            # 这样可以避免所有 batch 的梯度同时保留在 GPU 上
            with torch.no_grad():
                for p in actual_model.parameters():
                    if p.grad is not None:
                        p.grad = None

        # 3. 从 hook 中提取并聚合梯度（梯度已在 hook 中收集）
        base_grads = trainer.finalize_base_grads_collection()
        print(f"=====> base_grads collected from hooks: {len(base_grads)}")
        # 【关键修复 3】清理锚点梯度，确保不干扰主训练流程
        # 这一步在 ZeRO-2 下非常关键，防止下一次 optimizer.step() 把锚点梯度也算进去
        if ds_engine is not None:
            # 某些版本下可以使用 ds_engine.optimizer.zero_grad()
            # 最稳妥的是手动置 None
            for p in actual_model.parameters():
                p.grad = None
            # 强制重置 DeepSpeed 的内部梯度归约状态
            if hasattr(ds_engine, "is_gradient_accumulation_boundary"):
                # 这是一个 hack，防止引擎认为自己还处于某个累积周期中
                pass 
        else:
            actual_model.zero_grad(set_to_none=True)

        data_args.dataset = original_dataset 
        print(f"=====> base_grads length2: {len(base_grads)}")
        # base_grads 已经更新到 trainer.base_grads 中，这里返回用于兼容性
        return base_grads
    finally:
        # 恢复标志位
        if trainer is not None and hasattr(trainer, "_skip_gradient_projection"):
            trainer._skip_gradient_projection = False



def run_sft_pg(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    # print(f"=====> dataset_module train_dataset length: {len(dataset_module['train_dataset'])}=====>")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)


    # 创建data collator（用于后续计算base gradients）
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # 1. 准备通用知识的基准梯度 (Anchor Gradients)
    print("================================Base gradients computing.=================================")
    base_grads = None
    # 注意：初始 base_grads 计算需要在 trainer 创建之前，但我们现在要求必须传入 trainer
    # 所以先创建 trainer（base_grads=None），然后在 trainer 创建后计算 base_grads
    # 但这样会导致 trainer 初始化时 base_grads 为 None，hook 无法正常工作
    # 因此我们需要先创建一个临时的 trainer 来计算初始 base_grads
    # 或者修改逻辑，让初始计算也在 trainer 创建后进行
    
    # 注意：base_grads 的计算需要在 trainer 创建后进行，因为需要使用 hook 收集梯度
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # 2. 初始化自定义 Trainer
    print("================================OrthogonalProjectionTrainer initializing.=================================")
    # 2.1 构造 / 注入用于动态更新 base_grads 的回调
    #     - 如果外部已有 callbacks，则在其基础上 append
    effective_callbacks = list(callbacks) if callbacks is not None else []
    base_grad_callback = None
    if training_args.do_train and training_args.base_dataset is not None and getattr(
        training_args, "base_num_steps", None
    ):
        base_grad_callback = BaseGradUpdateCallback(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            tokenizer=tokenizer,
        )
        effective_callbacks.append(base_grad_callback)
        # effective_callbacks.append(ZeroOutGradientsCallback())

    trainer = OrthogonalProjectionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=effective_callbacks,
        gen_kwargs=gen_kwargs,
        base_grads=None,  # 初始为 None，会在计算后通过 hook 更新
        base_method=training_args.base_method if hasattr(training_args, 'base_method') else "orthogonal_projection",
        base_threshold=training_args.base_threshold if hasattr(training_args, 'base_threshold') else None,
        base_scale=training_args.base_scale if hasattr(training_args, 'base_scale') else 1.0,
        base_num_steps=training_args.base_num_steps if hasattr(training_args, 'base_num_steps') else 100000,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    # 将 trainer 引用注入回调
    if base_grad_callback is not None:
        base_grad_callback.trainer = trainer
    
    # 1. 准备通用知识的基准梯度 (Anchor Gradients) - 在 trainer 创建后计算
    print("================================Base gradients computing.=================================")
    if training_args.do_train and training_args.base_dataset is not None:
        logger.info_rank0(f"Computing base gradients with base_dataset={training_args.base_dataset}, base_num_samples={training_args.base_num_samples}")
        # 通过 hook 收集梯度，会自动更新到 trainer.base_grads
        compute_base_gradients(
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            trainer=trainer,  # 传入 trainer，梯度会在 hook 中自动收集并更新
        )
        logger.info_rank0(f"Base gradients computed. Number of parameters with base gradients: {len(trainer.base_grads) if trainer.base_grads else 0}")
    else:
        logger.info_rank0("Skipping base gradients computation (base_dataset is None or do_train is False)")
    print("================================Base gradients computed.=================================")
    print("================================OrthogonalProjectionTrainer initialized.=================================")

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
