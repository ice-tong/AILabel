# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from command import configs
from fairseq.criterions import FairseqCriterion, register_criterion
import random
from sklearn.metrics import average_precision_score

import numpy as np
np.seterr(divide='ignore',invalid='ignore')


def ml_collater(
    values,
    pad_idx=0,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


@register_criterion("ailabel")
class AILabelLoss(FairseqCriterion):
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
        ail_token = sample["net_input"]["src_tokens"][configs.static_field]
        all_targets_code = sample["target"]["tgt_tokens"][configs.static_field_label]

        mask_idx = self.task.source_dictionary[
            configs.static_field].indices["<mask>"]
        masked_code = ail_token.eq(mask_idx)
        if not masked_code.any():
            return self.last_result
        # if not masked_code.any():
        #     logging_output = {
        #         "loss": 0,
        #         'code_loss': 0,
        #         "ntokens": sample["ntokens"],
        #         "nsentences": sample["nsentences"],
        #         "sample_size": len(sample),
        #         "sample_size_code": len(sample),
        #         "accuracy": 0,
        #     }
        #     return 0, 0, logging_output

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
        
        # all_targets_code = ml_collater(
        #     all_targets_code, pad_to_length=ail_token.shape[1])
        targets_list = []
        for row_idx in range(ail_token.shape[0]):
            mask_num = torch.where(ail_token[row_idx]==mask_idx)[0].shape[0]
            for i in range(mask_num):
                targets_list.append(all_targets_code[row_idx][i])
        targets_code = torch.stack(targets_list)
        # targets_code = all_targets_code[masked_code]

        output, _ = model(**sample["net_input"], masked_code=masked_code,
            classification_head_name="maskvar")

        map = average_precision_score(
            targets_code.detach().cpu().numpy(),
            torch.sigmoid(output).detach().cpu().numpy(),
            average='samples')
        
        loss = 100 * F.binary_cross_entropy_with_logits(output, targets_code)
        
        sample_size_code = masked_code.int().sum()
        sample_size = sample_size_code

        if random.random() < 0.05:
            pred_label = (torch.sigmoid(output) > 0.5).int()
            for pred, label in zip(pred_label[:5], targets_code[:5]):
                print(
                    torch.where(pred.detach().cpu()==1)[0],
                    torch.where(label.detach().cpu()==1)[0]
                    )

        # if random.random() < 0.005 and sample_size > 15:  # only randomly log some prediction in case screen flushing

        #     idx = random.randint(0, sample_size-15)
        #     idx -= idx % 5

        #     targets_code_idx = targets_code.view(-1)[idx:idx+15]
        #     pred_x = pred_logits_code.view(-1, pred_logits_code.size(-1))[idx:idx+15]
        #     if pred_x.shape[0] != 0:
        #         pred_code_idx = torch.argmax(pred_x, dim=-1)
        #         print(f'tgt code:', self.task.source_dictionary[configs.static_field_label].string(targets_code_idx))
        #         print(f'pred code:', self.task.source_dictionary[configs.static_field_label].string(pred_code_idx))
        #     else:
        #         print(f'tgt code:', self.task.source_dictionary[configs.static_field_label].string(targets_code_idx))
        #         print(f'pred code: Nothing')

        logging_output = {
            "loss": loss.data,
            'code_loss': loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "sample_size_code": sample_size_code,
            "accuracy": map,
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
