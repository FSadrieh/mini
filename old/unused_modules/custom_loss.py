from torchtune.modules.loss import LinearCrossEntropyLoss

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()


class CustomLoss(LinearCrossEntropyLoss):
    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__(num_output_chunks=num_output_chunks, ignore_index=ignore_index)

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        sample_count: list[int],
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: loss tensor
        """
        bsz, seq_len = outputs.shape[:-1]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")

        logits = self.linear_projection(outputs)  # [bsz, seq_len, vocab_size]

        flat_logits = logits.view(-1, logits.size(-1))  # [bsz*seq_len, vocab]
        flat_targets = targets.view(-1)  # [bsz*seq_len]
        flat_losses = F.cross_entropy(
            flat_logits, flat_targets, reduction="none", ignore_index=self.ignore_index
        )  # [bsz*seq_len]

        losses = flat_losses.view(bsz, seq_len)  # [bsz, seq_len]
        mask = targets != self.ignore_index  # [bsz, seq_len]

        valid_per_example = mask.sum(dim=1).clamp(min=1).float()
        per_example_loss = losses.sum(dim=1) / valid_per_example  # [bsz]

        total_loss = (losses * mask).sum()
        total_valid = valid_per_example.sum()
        base_loss = total_loss / total_valid

        # Optional sanity-check against chunked implementation:
        with torch.no_grad():
            parent_loss = super().forward(outputs, targets)  # does the chunk/mask dance
            # allow a tiny fp tolerance:
            assert torch.allclose(
                base_loss, parent_loss, atol=1e-6
            ), f"Base loss differs: {base_loss.item():.8f} vs {parent_loss.item():.8f}"

        return base_loss, per_example_loss
