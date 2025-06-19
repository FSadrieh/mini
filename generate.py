import torch
from omegaconf import DictConfig

from torchtune import utils

from transformers import AutoModelForCausalLM, set_seed

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
        self.checkpoint_dir = None
        self._device = device
        self._dtype = dtype
        self._seed = seed

    def setup(self, checkpoint_dir, pad_token) -> bool:
        # We do not evaluate a epoch twice
        if self.checkpoint_dir == checkpoint_dir:
            logger.info(
                "Checkpoint directory is already set in the config, skipping generate."
            )
            return False

        set_seed(self._seed)
        self.checkpoint_dir = checkpoint_dir
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_dir, torch_dtype=self._dtype
        ).to(self._device)
        self.model.config.pad_token_id = pad_token
        self.pad_token = pad_token
        self.model.eval()
        return True

    @torch.inference_mode()
    def generate(self, prompts, attention_mask, max_len):
        return self.model.generate(
            input_ids=prompts,
            attention_mask=attention_mask,
            max_new_tokens=max_len,
            do_sample=self.cfg.generation.do_sample,
            temperature=self.cfg.generation.temperature,
            top_k=self.cfg.generation.top_k,
            pad_token_id=self.pad_token,
        )
