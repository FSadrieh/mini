from typing import Literal

import lightning as L
from print_on_steroids import logger
from torch.optim import AdamW
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.optimization import get_scheduler
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from evaluate import load
import datetime

import torch


class BasicLM(L.LightningModule):
    def __init__(
        self,
        args,
        generation_config=None,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])
        self.args = args
        # For generation we should set the padding side to left so we need a new tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.hf_model_name, use_fast=True, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = generation_config

        config = AutoConfig.from_pretrained(args.hf_model_name, return_dict=True)
        config.attn_implementation="flash_attention_2"
        self.model = AutoModelForCausalLM.from_pretrained(args.hf_model_name, config=config)
        self.loss = CrossEntropyLoss(ignore_index=-100, reduction='none')

        self.acc_metric = load("accuracy", experiment_id=str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
        self.squad_metric = load("squad", experiment_id=str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))

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
        input_check = torch.where(labels == -100, -100, labels)
        assert torch.all(labels == input_check), f"input_ids and labels do not match: {input_ids} != {input_check}"
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        custom_loss = self.calculate_loss(output.logits, batch["labels"], batch["sample_count"])
        self.log_dict(
            {
                "train/loss": output.loss,
                "train/mean_loss": custom_loss["mean"],
                "train/sum_loss": custom_loss["sum"],
                "train/distance_loss": custom_loss["distance"],
                "train/var_loss": custom_loss["variance"],
            },
            on_step=True,
            on_epoch=False,
            batch_size=batch["input_ids"].shape[0],
            sync_dist=True,
        )
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        loss = self.calculate_loss(output.logits, batch["labels"], batch["sample_count"])
        predictions = torch.argmax(output.logits[:, :-1, :].contiguous(), axis=-1).view(-1)
        labels = batch["labels"][:, 1:].contiguous().view(-1)
        mask = labels.ne(-100)
        accuracy = self.acc_metric.compute(predictions=predictions[mask], references=labels[mask])["accuracy"]
        # Does not work complety as of now. Fix generation_input_ids
        # if self.args.generate_in_validation:
        #     self.generate(batch)
        self.log_dict(
            {
                "val/loss": output.loss,
                "val/accuracy": accuracy,
                "val/mean_loss": loss["mean"],
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
        total_mean_loss = per_token_loss.sum() / labels.ne(-100).sum().float()
        per_token_loss = per_token_loss.view(labels.size(0), labels.size(1))
        per_example_loss = per_token_loss.sum(dim=1) / labels.ne(-100).sum(dim=1).float()
        per_group_loss = torch.split(per_example_loss, sample_count)
        return {
            "distance": sum(torch.abs(max(losses) - min(losses)) for losses in per_group_loss) / len(per_group_loss),
            "variance": sum(torch.var(losses) for losses in per_group_loss) / len(per_group_loss),
            "sum": sum(losses.sum() for losses in per_group_loss) / len(per_group_loss),
            "mean": total_mean_loss,
        }

    def generate(self, batch):
        """
            Generate while training to log EM and F1 scores.
        """
        #TODO: We need the right input ids, left padded, without labales probably have to create them also in the batch
        predictions = self.model.generate(
            input_ids=batch["generation_input_ids"],
            attention_mask=batch["attention_masks"],
            max_new_tokens=1, #labels.shape[1] - generation_input_ids.shape[1],
            **self.generation_config,
        )
        # Decode labels and predictions only where labels not -100
        predictions = torch.where(batch["labels"] == -100, self.tokenizer.pad_token_id, predictions)
        labels = torch.where(batch["labels"] == -100, self.tokenizer.pad_token_id, batch["labels"])
        pred_text = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Get it into the squad format
        squad_label = [{'id': str(i), 'answers': {'text': [text], 'answer_start': [0]}} for i, text in enumerate(label_text)]
        squad_pred = [{'id': str(i), 'prediction_text': text} for i, text in enumerate(pred_text)]
        metrics = self.squad_metric.compute(predictions=squad_pred, references=squad_label)
        self.log_dict(
            {
                "val/em": metrics["exact_match"],
                "val/f1": metrics["f1"],
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
        )


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
