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

import deepspeed
import contextlib
import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader
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


def compute_base_gradients(model, tokenizer, model_args, data_args, training_args, finetuning_args):
    """
    计算通用知识锚点梯度，解决 ZeRO-3 下权重维度报错问题。
    """
    if training_args.should_log:
        print(">>> 正在启动锚点梯度计算...")
    
    base_dataset = training_args.base_dataset
    base_num_samples = training_args.base_num_samples
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    current_rank = dist.get_rank() if is_distributed else 0

    # 1. 判定是否为 ZeRO-3 分片模式
    # 在 LlamaFactory 中，可以通过 training_args 的 deepspeed 配置判断
    is_zero3 = False
    if training_args.deepspeed:
        # 检查 ds_config 是否包含 stage 3
        import json
        with open(training_args.deepspeed, 'r') as f:
            ds_config = json.load(f)
            if ds_config.get("zero_optimization", {}).get("stage") == 3:
                is_zero3 = True
                if current_rank == 0:
                    print(">>> 检测到 ZeRO-3 开启，将应用参数聚合上下文。")

    # 2. 准备数据集
    original_dataset = data_args.dataset
    data_args.dataset = [base_dataset] if isinstance(base_dataset, str) else base_dataset

    tokenizer_module = {"tokenizer": tokenizer}
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    
    actual_model = getattr(model, "module", model)
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

    # 3. 计算梯度
    actual_model.train()
    actual_model.zero_grad()
    device = training_args.device
    num_batches = len(dataloader)

    # 核心修正：使用 GatheredParameters 包裹整个前向和后向过程
    # 这会强制把分片的 1-D 权重拼回 2-D 矩阵
    param_context = (
        deepspeed.zero.GatheredParameters(actual_model.parameters(), modifier_rank=None) 
        if is_zero3 else contextlib.nullcontext()
    )

    with param_context:
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = actual_model(**batch)
            loss = outputs.loss
            (loss / num_batches).backward()

    # 4. 提取并同步梯度
    base_grads = {}
    with torch.no_grad():
        # 这里同样需要在 context 下提取，确保 grad 也是完整的
        with param_context:
            for name, param in actual_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    g = param.grad.detach().clone()
                    
                    # 多卡环境手动聚合各卡的梯度
                    if is_distributed:
                        dist.all_reduce(g, op=dist.ReduceOp.SUM)
                        g = g / dist.get_world_size()
                    
                    norm = torch.norm(g)
                    if norm > 1e-10:
                        # 存入 CPU 节省显存
                        base_grads[name] = (g / (norm + 1e-10)).cpu()

    # 5. 清理现场
    actual_model.zero_grad()
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

    # Initialize our Trainer
    # trainer = CustomSeq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     finetuning_args=finetuning_args,
    #     data_collator=data_collator,
    #     callbacks=callbacks,
    #     gen_kwargs=gen_kwargs,
    #     **dataset_module,
    #     **tokenizer_module,
    #     **metric_module,
    # )
        # 2. 初始化自定义 Trainer
    print("================================OrthogonalProjectionTrainer initializing.=================================")
    trainer = OrthogonalProjectionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        base_grads=base_grads,
        base_method=training_args.base_method if hasattr(training_args, 'base_method') else "orthogonal_projection",
        base_threshold=training_args.base_threshold if hasattr(training_args, 'base_threshold') else None,
        base_scale=training_args.base_scale if hasattr(training_args, 'base_scale') else 1.0,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
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
