from typing import Literal

import lightning as L
from print_on_steroids import logger
from torch.optim import AdamW
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.optimization import get_scheduler
from torch.nn import CrossEntropyLoss

import torch


class BasicLM(L.LightningModule):
    def __init__(
        self,
        args,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])
        self.args = args
        config = AutoConfig.from_pretrained(args.hf_model_name, return_dict=True)
        self.model = AutoModelForCausalLM.from_pretrained(args.hf_model_name, config=config)
        self.loss = CrossEntropyLoss(ignore_index=-100, reduction='none')

        # if self.freeze > 0:
        #     logger.info(f"Freezing {self.freeze} layers of the model")
        #     for param in self.model.parameters():
        #         param.requires_grad = False
        #     for i, layer in enumerate(self.model.encoder.layer):
        #         if i >= self.freeze:
        #             break
        #         for param in layer.parameters():
        #             param.requires_grad = True

    def forward(self, input_ids, attention_masks, labels, sample_count):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
        )

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = self.calculate_loss(output.logits, batch["labels"], batch["sample_count"])
        loss = loss[self.args.loss_type]
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=batch["input_ids"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        loss = self.calculate_loss(output.logits, batch["labels"], batch["sample_count"])
        self.log_dict(
            {
                "val/sum_loss": loss["sum"],
                "val/distance_loss": loss["distance"],
                "val/var_loss": loss["variance"],
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
        )

    def calculate_loss(self, logits, labels, sample_count):
        labels = labels[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        per_token_loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        per_token_loss = per_token_loss.view(labels.size(0), labels.size(1))
        per_example_loss = per_token_loss.sum(dim=1) / labels.ne(-100).sum(dim=1).float()
        per_group_loss = torch.split(per_example_loss, sample_count)
        return {
            "distance": sum(torch.abs(max(losses) - min(losses)) for losses in per_group_loss) / len(per_group_loss),
            "variance": sum(torch.var(losses) for losses in per_group_loss) / len(per_group_loss),
            "sum": sum(losses.sum() for losses in per_group_loss) / len(per_group_loss),
        }



    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay} and warmup steps: {self.args.warmup_period}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.epsilon,  # You can also tune this
        )

        scheduler_name = self.args.lr_schedule
        if scheduler_name == "constant" and self.args.warmup_period > 0:
            scheduler_name += "_with_warmup"
        scheduler = get_scheduler(
            scheduler_name,
            optimizer,
            num_warmup_steps=int(self.args.warmup_period),
            num_training_steps=self.trainer.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
