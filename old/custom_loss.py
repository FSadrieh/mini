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

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]

        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk

        Raises:
            AttributeError: if called before update_model
        """
        # Select hidden states and targets where mask is True
        mask_chunk = target_chunk != self.ignore_index
        if mask_chunk.sum() == 0:
            # Unmask 1 token to allow loss to sync with all data parallel workers
            mask_chunk[0] = True

        target_chunk = target_chunk[mask_chunk]  # [num_valid]
        if isinstance(hidden_chunk, DTensor):
            # DTensor doesn't support masks so we have to mask locally
            mesh = hidden_chunk.device_mesh
            placements = hidden_chunk.placements
            local_hidden_chunk = hidden_chunk.to_local()[mask_chunk]
            hidden_chunk = DTensor.from_local(
                local_hidden_chunk, mesh, placements
            )  # [num_valid, embed_dim]
        else:
            hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        return F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="none",
            ignore_index=self.ignore_index,
        )

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
        # Total number of non-ignored tokens across the entire batch
        mask = targets != self.ignore_index
        total_elements = mask.sum()

        # Chunk along sequence dimension
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)

        # Compute cross-entropy loss for the chunks
        total_losses = []
        for idx in range(len(hidden_chunks)):
            total_losses.append(self.compute_cross_entropy(
                hidden_chunks[idx],
                target_chunks[idx],
            ))

        per_token_loss = torch.cat(total_losses, dim=0) #TODO: Which shape
        per_example_loss = per_token_loss.sum(dim=1) / targets.ne(-100).sum(dim=1).float()
        per_group_loss = torch.split(per_example_loss, sample_count)
        return {
            "distance": sum(torch.abs(max(losses) - min(losses)) for losses in per_group_loss) / len(per_group_loss),
            "variance": sum(torch.var(losses) for losses in per_group_loss) / len(per_group_loss),
        }
