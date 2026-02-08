# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

from copy import deepcopy
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union, Dict
import deepspeed
import contextlib
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


class OrthogonalProjectionTrainer(CustomSeq2SeqTrainer):
    def __init__(
        self,
        *args,
        base_grads=None,
        base_method: str = "orthogonal_projection",
        base_threshold=None,
        base_scale: float = 1.0,
        base_num_steps: Optional[int] = None,
        **kwargs,
    ):
        """
        :param base_grads:      预先计算好的锚点梯度（参数名 -> Tensor on CPU）
        :param base_method:     "orthogonal_projection" 或 "threshold"
        :param base_threshold:  目前未使用，保留接口以便后续扩展
        :param base_scale:      在 "threshold" 策略下，对同向梯度缩放系数
        :param base_num_steps:  每隔多少个 global_step 期望更新一次 base_grads（真正的更新逻辑在外部回调里完成）
        """
        super().__init__(*args, **kwargs)
        self.base_grads = base_grads
        self.base_method = base_method
        self.base_threshold = base_threshold
        self.base_scale = base_scale
        self.base_num_steps = base_num_steps
        self.eps = 1e-8
        # 标志位：计算 base grad 时跳过梯度矫正
        self._skip_gradient_projection = False
        # 用于在 hook 中收集 base_grads 的临时字典
        self._collected_base_grads = {}
        # 期望收集的梯度数量（用于验证是否收集完整）
        self._expected_grad_count = 0
        
        # Register hooks during initialization
        # 即使 base_grads 为 None，也注册 hook（用于后续收集梯度）
        self._register_gradient_hooks()

    def _register_gradient_hooks(self):
        model = getattr(self.model, "module", self.model)
        
        def make_hook(name):
            def hook(grad):
                if grad is None:
                    return grad
                
                # 如果正在计算 base grad，收集梯度并跳过梯度矫正
                if self._skip_gradient_projection:
                    # 在 hook 中收集梯度，用于更新 base_grads
                    # 注意：这里收集的是原始梯度，不做任何处理
                    if name not in self._collected_base_grads:
                        # 第一个 batch：克隆梯度并立即转移到 CPU，节省 GPU 内存
                        # 注意：这里先 detach 再 clone，避免保留计算图
                        self._collected_base_grads[name] = grad.detach().clone().cpu()
                        # print(f"[Hook] Collected gradient for {name}, shape: {grad.shape}")
                    else:
                        # 后续 batch：累加梯度（多个 batch 的情况）
                        # 优化：直接在 CPU 上累加，避免 GPU 内存占用
                        # 注意：grad 在 GPU 上，需要先转移到 CPU 再累加
                        grad_cpu = grad.detach().clone().cpu()
                        self._collected_base_grads[name] = self._collected_base_grads[name] + grad_cpu
                        del grad_cpu  # 立即释放内存
                    return grad
                
                # 动态获取 base_grad（因为 base_grads 可能在注册后更新）
                if self.base_grads is None or name not in self.base_grads:
                    return grad
                
                base_grad_cpu = self.base_grads[name]
                
                # ZeRO-2 下，grad 的 shape 始终与模型定义的 shape 一致
                g_base = base_grad_cpu.to(device=grad.device, dtype=grad.dtype)
                # print(f"=====> g_base: {g_base}")

                 # --- 计算原始夹角 ---
                def get_angle(v1, v2):
                    cos_sim = torch.sum(v1 * v2) / (torch.norm(v1) * torch.norm(v2) + self.eps)
                    # 限制在 [-1, 1] 范围内防止 acos 报错
                    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                    angle_rad = torch.acos(cos_sim)
                    return torch.rad2deg(angle_rad).item()

                # 只在 rank 0 且特定步数打印，避免性能损失和日志冗余
                # should_log = (torch.distributed.get_rank() == 0 and 
                #              getattr(self.state, "global_step", 0) % self.args.logging_steps == 0)
                # should_log = True
                should_log = False
                
                if should_log:
                    angle_before = get_angle(grad, g_base) if should_log else None

                # 执行投影逻辑 (保持不变)
                if self.base_method == "orthogonal_projection":
                    dot_product = torch.sum(grad * g_base)
                    norm_base_sq = torch.sum(g_base * g_base) + self.eps
                    grad.sub_((dot_product / norm_base_sq) * g_base)
                elif self.base_method == "threshold":
                    dot_product = torch.sum(grad * g_base)
                    norm_grad = torch.norm(grad) + self.eps
                    norm_base = torch.norm(g_base) + self.eps
                    cos_theta = dot_product / (norm_grad * norm_base)     
                    if cos_theta < 0:
                        norm_base_sq = torch.sum(g_base * g_base) + self.eps
                        grad.sub_((dot_product / norm_base_sq) * g_base)
                    elif cos_theta > 0:
                        grad.mul_(self.base_scale)

                # --- 计算投影后夹角并打印 ---
                if should_log:
                    angle_after = get_angle(grad, g_base)
                    # 选取一些关键层打印，防止每一层都输出
                    if "layers.0" in name or "layers.31" in name or "embed_tokens" in name:
                        print(f"[GradProj] Step: {self.state.global_step} | Param: {name}")
                        print(f"Angle Before: {angle_before:.2f}° | After: {angle_after:.2f}°")
                
                return grad
            return hook

        for name, param in model.named_parameters():
            if param.requires_grad:
                # 注册 hook，base_grad 会在 hook 内部动态获取（避免闭包问题）
                param.register_hook(make_hook(name))

    def start_base_grads_collection(self):
        """开始收集 base_grads，清空之前的收集结果"""
        self._collected_base_grads = {}
        model = getattr(self.model, "module", self.model)
        # 计算期望收集的梯度数量（统计所有 requires_grad 的参数，因为所有参数都注册了 hook）
        self._expected_grad_count = sum(
            1 for name, param in model.named_parameters()
            if param.requires_grad
        )
        print(f"[BaseGrads] Started collection, expecting {self._expected_grad_count} gradients")
    
    def finalize_base_grads_collection(self):
        """
        完成 base_grads 的收集，处理收集到的梯度（聚合、归一化）并更新 base_grads
        返回更新后的 base_grads 字典
        """
        import torch.distributed as dist
        
        if not self._collected_base_grads:
            print("[BaseGrads] Warning: No gradients collected")
            return self.base_grads
        
        print(f"[BaseGrads] Collected {len(self._collected_base_grads)} gradients, expected {self._expected_grad_count}")
        
        new_base_grads = {}
        # 注意：收集的梯度已经在 CPU 上了，不需要再获取 device
        
        for name, grad in self._collected_base_grads.items():
            if grad is None:
                continue
            
            # 1. 聚合梯度（如果是分布式训练）
            # 需要先转移到 GPU 进行 all_reduce，然后再转回 CPU

            if dist.is_initialized():
                # 获取一个参考设备（通常是第一个参数的设备）
                if not hasattr(self, '_reference_device'):
                    model = getattr(self.model, "module", self.model)
                    self._reference_device = next(model.parameters()).device
                
                grad_gpu = grad.to(device=self._reference_device)
                dist.all_reduce(grad_gpu, op=dist.ReduceOp.SUM)
                grad_gpu = grad_gpu / dist.get_world_size()
                grad = grad_gpu.cpu()  # 立即转回 CPU
                del grad_gpu  # 释放 GPU 内存
            
            # 2. 归一化并存储到 CPU（已经在 CPU 上）
            norm = torch.norm(grad)
            if norm > 1e-10:
                normalized_grad = grad / (norm + 1e-10)
                new_base_grads[name] = normalized_grad  # 已经在 CPU 上
        
        # 3. 更新 base_grads
        if new_base_grads:
            print(f"[BaseGrads] Updated {len(new_base_grads)} base gradients")
            self.base_grads = new_base_grads
            # print(f'=====> new_grads:{self.base_grads}')
            # 清空收集的梯度，释放内存
            self._collected_base_grads = {}
        
        return self.base_grads