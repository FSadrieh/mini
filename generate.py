# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.data import Message, Role
from torchtune.training import FullModelTorchTuneCheckpointer

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig, device, dtype, seed) -> None:
        self.cfg = cfg
        self._device = device
        self._dtype = dtype

        training.set_seed(
            seed=seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self._tokenizer = None

    def setup(self, checkpoint_dir) -> bool:
        # We do not evaluate a epoch twice
        if self.cfg.checkpointer.checkpoint_dir == checkpoint_dir and "epoch" in checkpoint_dir:
            logger.info(
                "Checkpoint directory is already set in the config, skipping setup."
            )
            return False
        self.cfg.checkpointer.checkpoint_dir = checkpoint_dir
        checkpointer = config.instantiate(self.cfg.checkpointer)
        ckpt_dict = checkpointer.load_checkpoint()

        with training.set_default_dtype(self._dtype), self._device:
            self._model = config.instantiate(self.cfg.model)

        self._model.load_state_dict(ckpt_dict[training.MODEL_KEY],)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            self._model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")
        return True

    @torch.inference_mode()
    def generate(self, prompts, max_len):

        # Ensure the cache is setup on the right device, with only as many tokens as we need
        #TODO: KV cache breaks wiht more than one batch, either it is faster 
        # if self.cfg.enable_kv_cache:
        #     with self._device:
        #         self._model.setup_caches(
        #             batch_size=prompts.size(0),
        #             dtype=self._dtype,
        #             decoder_max_seq_len=prompts.size(1) + max_len,
        #         )

        generated_tokens, _ = generation.generate(
            model=self._model,
            prompt=prompts,
            max_generated_tokens=max_len,
            pad_id=self._tokenizer.pad_id,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
        )
        return generated_tokens



@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
