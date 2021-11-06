# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from commands import configs
from fairseq.criterions import FairseqCriterion, register_criterion
import random


@register_criterion("ail_trace")
class AILTraceLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        mask_idx = self.task.source_dictionary[
            configs.static_field].indices["<mask>"]
        vpad_idx = self.task.source_dictionary[
            configs.static_field_label].indices["<vpad>"]
        masked_code = sample["net_input"]["src_tokens"]["ail_token"].eq(mask_idx)

        if not masked_code.any():
            return self.last_result

        # Rare: when all tokens are not masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_code = None  # always project all tokens on TPU
        elif masked_code.device == torch.device("cpu"):
            if not masked_code.any():
                masked_code = None
        else:
            masked_code = torch.where(
                masked_code.any(),
                masked_code,
                masked_code.new([True]),
            )

        output, _ = model(**sample["net_input"], masked_code=masked_code,
            classification_head_name="maskvar")

        pred_logits_code = output
        pred_code = torch.argmax(pred_logits_code, dim=-1)
        targets_code = sample["target"]["tgt_tokens"]

        if masked_code is not None:
            targets_code = targets_code[configs.static_field_label][masked_code]
        
        remove_vpad_acc = (targets_code == pred_code) & targets_code.eq(vpad_idx)
        acc = torch.mean((pred_code == targets_code).float()[~remove_vpad_acc])
        

        sample_size_code = masked_code.int().sum()
        sample_size = sample_size_code

        code_loss = modules.cross_entropy(
            pred_logits_code.view(-1, pred_logits_code.size(-1)),
            targets_code.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx_dict[configs.static_field],
            # ignore_index=self.task.source_dictionary[configs.static_field_label].indices["<vpad>"],
        )

        loss = code_loss

        if random.random() < 0.005 and sample_size > 15:  # only randomly log some prediction in case screen flushing

            idx = random.randint(0, sample_size-15)
            idx -= idx % 5

            targets_code_idx = targets_code.view(-1)[idx:idx+15]
            pred_x = pred_logits_code.view(-1, pred_logits_code.size(-1))[idx:idx+15]
            if pred_x.shape[0] != 0:
                pred_code_idx = torch.argmax(pred_x, dim=-1)
                print(f'tgt code:', self.task.source_dictionary[configs.static_field_label].string(targets_code_idx))
                print(f'pred code:', self.task.source_dictionary[configs.static_field_label].string(pred_code_idx))
            else:
                print(f'tgt code:', self.task.source_dictionary[configs.static_field_label].string(targets_code_idx))
                print(f'pred code: Nothing')

        logging_output = {
            "loss": loss.data,
            'code_loss': code_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "sample_size_code": sample_size_code,
            "accuracy": acc,
        }
        self.last_result = loss, sample_size, logging_output
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        code_loss_sum = sum(log.get("code_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_code = sum(log.get("sample_size_code", 0) for log in logging_outputs)
        acc_list = [log.get("accuracy", 0) for log in logging_outputs]
        # acc_list = [i for i in acc_list if i != 0]
        acc = sum(acc_list) / len(acc_list)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("code_loss", code_loss_sum / sample_size_code / math.log(2), sample_size_code, round=3)
        metrics.log_derived("code_ppl", lambda meters: utils.get_perplexity(meters["code_loss"].avg))
        metrics.log_scalar("accuracy", sum(acc_list) / len(acc_list), 1, round=2)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
