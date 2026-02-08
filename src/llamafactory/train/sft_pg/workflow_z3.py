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
        if trainer is None or not hasattr(trainer, "update_base_grads"):
            return control

        # 重新计算 base_grads（使用和初始阶段同样的数据与逻辑）
        # 注意：在训练循环中调用时，compute_base_gradients 会处理 ZeRO-3 的状态清理
        new_base_grads = compute_base_gradients(
            model=model,
            tokenizer=self.tokenizer,
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=self.training_args,
            finetuning_args=self.finetuning_args,
        )

        # 更新到 Trainer 中，内部会自动让已注册的 hook 使用新的 base_grads
        trainer.update_base_grads(new_base_grads)

        if args.should_log:
            logger.info_rank0(
                f">>> Step {state.global_step}: base_grads updated. "
                f"有效参数数目: {len(new_base_grads) if new_base_grads else 0}"
            )

        return control




def compute_base_gradients(model, tokenizer, model_args, data_args, training_args, finetuning_args):
    """
    更安全的锚点梯度计算：
    - 前/后向优先通过 DeepSpeedEngine（如存在），避免直接调用被分片的 module 触发 shape/active_sub_modules 问题。
    - 仅在提取梯度时使用 GatheredParameters；退出前清理梯度，保持训练主循环状态干净。
    """
    if training_args.should_log:
        print(">>> 正在启动锚点梯度计算...")
    
    base_dataset = training_args.base_dataset
    base_num_samples = training_args.base_num_samples
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    current_rank = dist.get_rank() if is_distributed else 0

    # 判定 ZeRO-3
    is_zero3 = False
    if training_args.deepspeed:
        import json
        with open(training_args.deepspeed, "r") as f:
            ds_config = json.load(f)
            if ds_config.get("zero_optimization", {}).get("stage") == 3:
                is_zero3 = True
                if current_rank == 0:
                    print(">>> 检测到 ZeRO-3 开启，将使用 DS Engine 执行前后向。")

    # 备份并切换数据集
    original_dataset = data_args.dataset
    data_args.dataset = [base_dataset] if isinstance(base_dataset, str) else base_dataset

    tokenizer_module = {"tokenizer": tokenizer}
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    
    actual_model = getattr(model, "module", model)
    ds_engine = model if hasattr(model, "backward") and hasattr(model, "module") else None
    model_config = getattr(actual_model, "config", None)
    
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=actual_model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model_config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    train_dataset = dataset_module.get("train_dataset")
    num_samples_to_select = min(base_num_samples, len(train_dataset))
    dataloader = DataLoader(
        train_dataset.select(range(num_samples_to_select)), 
        batch_size=training_args.per_device_train_batch_size, 
        collate_fn=data_collator,
        shuffle=False
    )

    device = training_args.device
    num_batches = max(len(dataloader), 1)

    base_grads: dict[str, torch.Tensor] = {}
    params_for_gather = (
        list(ds_engine.module.parameters()) if ds_engine is not None else list(actual_model.parameters())
    )

    try:
        # 前/后向：优先走 DS Engine
        # 前/后向上下文：若 ZeRO-3 但还未有 ds_engine，就用 GatheredParameters 保证权重为 2-D
        forward_ctx = (
            contextlib.nullcontext()
            if ds_engine is not None
            else (deepspeed.zero.GatheredParameters(params_for_gather, modifier_rank=None) if is_zero3 else contextlib.nullcontext())
        )

        if ds_engine is not None:
            ds_engine.train()
            (ds_engine.optimizer or ds_engine).zero_grad(set_to_none=True)  # type: ignore
        else:
            actual_model.train()
            actual_model.zero_grad(set_to_none=True)

        with forward_ctx:
            for batch in tqdm(dataloader, desc="Computing base gradients"):
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                loss = (ds_engine(**batch).loss if ds_engine is not None else actual_model(**batch).loss) / num_batches

                if ds_engine is not None:
                    ds_engine.backward(loss)
                else:
                    loss.backward()

        # 只在提取梯度时 gather
        gather_ctx = (
            deepspeed.zero.GatheredParameters(params_for_gather, modifier_rank=None)
            if is_zero3
            else contextlib.nullcontext()
        )
        with gather_ctx:
            with torch.no_grad():
                for name, param in actual_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        g = param.grad.detach()
                        if is_distributed:
                            dist.all_reduce(g, op=dist.ReduceOp.SUM)
                            g = g / dist.get_world_size()

                        norm = torch.norm(g)
                        if norm > 1e-10:
                            base_grads[name] = (g / (norm + 1e-10)).cpu()

        # 清理梯度，避免污染主训练
        if ds_engine is not None:
            if getattr(ds_engine, "optimizer", None) is not None:
                ds_engine.optimizer.zero_grad(set_to_none=True)
            else:
                ds_engine.zero_grad(set_to_none=True)  # type: ignore
        else:
            actual_model.zero_grad(set_to_none=True)

        if is_distributed:
            dist.barrier()

    finally:
        data_args.dataset = original_dataset

    if current_rank == 0:
        logger.info(f">>> 锚点计算完成。有效参数: {len(base_grads)}")

    return base_grads



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
    print(f"dataset_module train_dataset length: {len(dataset_module['train_dataset'])}")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    # whether model is a nn.Module or a nn.DataParallel or deepspeed.zero.GatheredParameters
    if isinstance(model, torch.nn.parallel.DataParallel):
        print(f"=====> model is a nn.DataParallel")
    elif isinstance(model, deepspeed.zero.GatheredParameters):
        print(f"=====> model is a deepspeed.zero.GatheredParameters")
    else:
        print(f"=====> model is a nn.Module")


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
    if training_args.do_train and training_args.base_dataset is not None:
        logger.info_rank0(f"Computing base gradients with base_dataset={training_args.base_dataset}, base_num_samples={training_args.base_num_samples}")
        base_grads = compute_base_gradients(
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
        )
        logger.info_rank0(f"Base gradients computed. Number of parameters with base gradients: {len(base_grads) if base_grads else 0}")
    else:
        logger.info_rank0("Skipping base gradients computation (base_dataset is None or do_train is False)")
    print("================================Base gradients computed.=================================")
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

    trainer = OrthogonalProjectionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=effective_callbacks,
        gen_kwargs=gen_kwargs,
        base_grads=base_grads,
        base_method=training_args.base_method if hasattr(training_args, 'base_method') else "orthogonal_projection",
        base_threshold=training_args.base_threshold if hasattr(training_args, 'base_threshold') else None,
        base_scale=training_args.base_scale if hasattr(training_args, 'base_scale') else 1.0,
        base_num_steps=training_args.base_num_steps if hasattr(training_args, 'base_num_steps') else 100000,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    # 将 trainer 引用注入回调，方便在回调内部调用 update_base_grads
    if base_grad_callback is not None:
        base_grad_callback.trainer = trainer
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
