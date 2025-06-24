import torch


def calculate_loss(custom_loss, logits, labels, sample_count):
    per_token_loss = custom_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
    total_mean_loss = per_token_loss.sum() / labels.ne(-100).sum().float()
    per_token_loss = per_token_loss.view(labels.size(0), labels.size(1))
    per_example_loss = per_token_loss.sum(dim=1) / labels.ne(-100).sum(dim=1).float()
    per_group_loss = torch.split(per_example_loss, sample_count)
    return {
        "distance": sum(
            torch.abs(max(losses) - min(losses)) for losses in per_group_loss
        )
        / len(per_group_loss),
        "variance": sum(torch.var(losses, unbiased=False) for losses in per_group_loss)
        / len(per_group_loss),
        "sum": sum(losses.sum() for losses in per_group_loss) / len(per_group_loss),
        "mean": total_mean_loss,
    }


def log_metrics_over_epoch(
    new_metrics: dict[str, float],
    running_metrics: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    for key, value in new_metrics.items():
        if key not in running_metrics:
            running_metrics[key] = torch.tensor(0.0, device=device)
        if key == "current_num_tokens":
            running_metrics[key] += value
        else:
            running_metrics[key] += (
                value.detach().float() * new_metrics["current_num_tokens"]
            )

    return running_metrics


def calculate_total_metrics(metrics: dict[str, float]) -> dict[str, float]:
    total_metrics = {}
    for key, value in metrics.items():
        if key == "current_num_tokens":
            total_metrics[key] = value
            continue
        total_metrics[key] = (
            value / metrics["current_num_tokens"]
            if metrics["current_num_tokens"] > 0
            else float("inf")
        )
    return total_metrics


def make_sanity_check(
    prompts: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    position_ids: torch.Tensor,
    generator,
    tokenizer,
):
    def _assertion(pred_1, pred_2, msg):
        if not (pred_1 == pred_2).all():
            for i, pred in enumerate(pred_1):
                if not (pred == pred_2[i]).all():
                    return True, f"{msg} at index {i}: {pred_1[i]} != {pred_2[i]}"
        return False, ""

    # We generate the predictions again to ensure determinism
    predictions_again = generator.generate(
        prompts=prompts,
        attention_mask=attention_mask,
        max_new_tokens=labels.shape[1],
        position_ids=position_ids,
    )

    error, msg = _assertion(
        predictions,
        predictions_again,
        "Predictions with position_ids differ from the original",
    )
    if error:
        return error, msg

    # Check if position_ids make a difference
    predictions_no_pids = generator.generate(
        prompts=prompts,
        attention_mask=attention_mask,
        max_new_tokens=labels.shape[1],
        position_ids=None,  # No position_ids
    )
    _assertion(
        predictions,
        predictions_no_pids,
        "Predictions with position_ids differ from the original without position_ids",
    )
    if error:
        return error, msg

    # We generate unbatched predictions to enusre our batching does not change anything
    individual_predictions = []
    for prompt in prompts:
        prompt = prompt[prompt.ne(tokenizer.pad_id)].unsqueeze(0)
        attention_mask = torch.ones_like(prompt, dtype=torch.long)
        individual_predictions.append(
            generator.generate(
                prompts=prompt,
                attention_mask=attention_mask,
                max_new_tokens=labels.shape[1],
                position_ids=None,
            )
        )

    # Make one tensor through stacking and padding
    padded = torch.stack(
        [
            torch.nn.functional.pad(
                t, (0, labels.shape[1] - t.shape[1]), value=tokenizer.eos_id
            )
            for t in individual_predictions
        ]
    ).squeeze(1)
    _assertion(
        predictions,
        padded,
        "Predictions with position_ids differ from the unbatched generation",
    )
    if error:
        return error, msg
    return False, "Sanity check passed, all predictions match expected results."


def get_em_per_prompt(
    squad_pred: list[str],
    squad_label: list[str],
    templates: list[str],
    metric,
    em_per_prompt: dict[str, list[float]],
):
    for i, prompt_id in enumerate(templates):
        metric.reset()
        metric.update(preds=[squad_pred[i]], target=[squad_label[i]])
        if prompt_id not in em_per_prompt:
            em_per_prompt[prompt_id] = []
        em_per_prompt[prompt_id].append(metric.compute()["exact_match"].item())
    return em_per_prompt
