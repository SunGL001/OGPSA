# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

from typing import TYPE_CHECKING, Optional

import deepspeed
import contextlib
import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader
from ...data import (
    PairwiseDataCollatorWithPadding,
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer, OrthogonalProjectionDPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


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

    
    # tokenizer_module = load_tokenizer(model_args)
    # tokenizer = tokenizer_module["tokenizer"]
    # template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    # model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    tokenizer_module = {"tokenizer": tokenizer}
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    
    actual_model = getattr(model, "module", model)
    model_config = getattr(actual_model, "config", None)
    
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=actual_model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
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


def run_dpo_pg(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
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

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # 2. 初始化自定义 Trainer
    print("================================OrthogonalProjectionDPOTrainer initializing.=================================")
    trainer = OrthogonalProjectionDPOTrainer(
        model=model,
        ref_model=ref_model,
        finetuning_args=finetuning_args,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks,
        base_grads=base_grads,
        base_method=training_args.base_method if hasattr(training_args, 'base_method') else "orthogonal_projection",
        base_threshold=training_args.base_threshold if hasattr(training_args, 'base_threshold') else None,
        base_scale=training_args.base_scale if hasattr(training_args, 'base_scale') else 1.0,
        **dataset_module,
        **tokenizer_module,
    )
    print("================================OrthogonalProjectionDPOTrainer initialized.=================================")

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="rm"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss", "rewards/accuracies"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
