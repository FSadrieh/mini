    def get_inner_validate(self) -> dict[str, float]:
        """
        Returns the handling for a single validation batch depending if they are:
        - template_batches (all templates for one sample are in the batch)
        - data_batches (a batch contains a batch for each template, these inner batches contain only one template for multiple examples)
        """

        def _get_additional_metrics(
            metrics: dict[str, float],
            batch: dict[str, torch.Tensor],
            batch_idx: int,
            logits: torch.Tensor,
            templates: list[str],
        ) -> dict[str, float]:
            metrics["current_num_tokens"] = (
                batch["labels"] != self._loss_fn.ignore_index
            ).sum()

            self.ppl.update(
                logits,
                batch["labels"],
            )
            predictions = torch.argmax(logits, axis=-1).view(-1)
            mask = batch["labels"].view(-1).ne(-100)
            self.acc.update(
                preds=predictions[mask],
                target=batch["labels"].view(-1)[mask],
            )

            self.generate(batch, templates, batch_idx)

            return metrics

        def _inner_validate_template_batches(
            batch_idx: int,
            batch: dict[str, torch.Tensor],
            running_metrics: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            templates = batch.pop("templates", None)
            utils.batch_to_device(batch, self._device)

            # Compute loss
            loss, logits = self._loss_step(batch)

            metrics = calculate_custom_losses(
                self.custom_loss,
                logits,
                batch["labels"],
                batch["sample_count"].tolist(),
            )
            metrics["loss"] = loss

            metrics = _get_additional_metrics(
                metrics, batch, batch_idx, logits, templates
            )

            return log_metrics_over_epoch(metrics, running_metrics, self._device)

        def _inner_validate_data_batches(
            dataset_batch_idx: int,
            dataset_batch: dict[str, dict[str, torch.Tensor]],
            running_metrics: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            template_metrics = []

            for template_batch_idx, template_batch in enumerate(dataset_batch):
                templates = template_batch.pop("templates", None)
                utils.batch_to_device(template_batch, self._device)

                # Compute loss
                loss, logits = self._loss_step(template_batch)
                metrics = {"loss": loss}
                metrics = _get_additional_metrics(
                    metrics, template_batch, template_batch_idx, logits, templates
                )
                template_metrics.append(metrics)

            dataset_metrics = calculate_custom_data_batch_losses(template_metrics)
            return log_metrics_over_epoch(
                dataset_metrics, running_metrics, self._device
            )

        return (
            _inner_validate_data_batches
            if self.is_data_batches
            else _inner_validate_template_batches
        )

    def validate(self) -> dict[str, float]:
        """
        Run validation loop and return average validation loss.
        """
        self._model.eval()
        for m in (self.acc, self.bleu, self.squad, self.rouge, self.ppl):
            m.reset()
        running_metrics = {}
        self.em_per_prompt = {}

        self.generator.setup(  # TODO: SEE ERROR_AVR.TXT
            self.current_checkpoint_dir,
            self._tokenizer.pad_id,
            [self._tokenizer.eos_id, self._tokenizer.special_tokens["<|eot_id|>"]],
        )

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self._val_dataloader),
                disable=not self._is_rank_zero,
                desc="Validating",
                total=self._eval_batches if self._eval_batches > 0 else None,
            ):
                if batch_idx >= self._eval_batches and self._eval_batches > 0:
                    break

                running_metrics = self.inner_validate(batch_idx, batch, running_metrics)

        # Aggregate validation metrics across all ranks
        metrics = calculate_total_metrics(running_metrics)
        squad_metrics = self.squad.compute()
        # Validation metrics
        log_dict = {
            "val/accuracy": self.acc.compute().item(),
            "val/loss": metrics["loss"],
            "val/mean_loss": metrics["mean"],
            "val/sum_loss": metrics["sum"],
            "val/distance_loss": metrics["distance"],
            "val/var_loss": metrics["variance"],
            "val/ppl": self.ppl.compute().item(),
        }

        # Generation metrics
        log_dict.update(
            {
                "val/exact_match": squad_metrics["exact_match"].item(),
                "val/f1": squad_metrics["f1"].item(),
                "val/bleu": self.bleu.compute().item(),
                "val/rougeL": self.rouge.compute()["rougeL_fmeasure"].item(),
            }
        )

        for prompt, em in self.em_per_prompt.items():
            em = torch.tensor(em)
            num_samples = em.shape[0]
            log_dict[f"em/{prompt}"] = (
                (em.sum() / num_samples) if num_samples > 0 else 0.0
            )

        self._logger.info(f"Validation loss: {metrics['loss']:.4f}")
        if self._is_rank_zero:
            self._logger.info(log_dict)
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )

        self._model.train()
        return log_dict