import torch


def validate_template_batches(
    batch: dict[str, torch.Tensor | list[str]], train_max_batch_size: int, val_max_batch_size: int
):
    """The idea of a template_batch is to pack all prompts for a sample into a single batch and then pack the batches with as many samples as possible.
    A valid template batch is a dictionary with the following keys:
    - 'generation_label': a tensor of shape (batch_size, n) with generation labels
    - 'labels': a tensor of shape (batch_size, n) with labels
    - 'prompt': a tensor of shape (batch_size, m) with prompts
    - 'sample_count': a tensor of shape (batch_size,) with sample counts
    - 'templates': a list of strings with templates
    - 'tokens': a tensor of shape (batch_size, p) with tokens
    NOTE: The batch size will vary depending how many samples we can pack into a batch.

    The function checks the following:
    1. Validate the basic of the template batch
    2. Validate batch lengths
    3. Validate individual contents of the batch
    """
    _validate_basic_batch(batch, train_max_batch_size, val_max_batch_size, is_template_batch=True)


def validate_data_batches(
    batches: list[dict[str, torch.Tensor | list[str]]], train_batch_size: int, val_batch_size: int
):
    """The idea of a data_batch is to have a list of batches. In each of the batches we have only one template for different samples. Over all batches in the list we have all templates.
    A valid template batch is a list of dictionaries with the following keys:
    - 'generation_label': a tensor of shape (batch_size, n) with generation labels
    - 'labels': a tensor of shape (batch_size, n) with labels
    - 'prompt': a tensor of shape (batch_size, m) with prompts
    - 'templates': a list of strings with templates
    - 'tokens': a tensor of shape (batch_size, p) with tokens
    NOTE: The length of the list will vary based on how many templates we have, but the batch size will be the same for all batches.

    The function checks the following:
    1. Validate the basic of the template batch
    2. Validate batch lengths
    3. Validate individual contents of the batch
    """
    ### ASSERT GENERAL STRUCTURE ###
    assert isinstance(batches, list), f"batch should be a dict, but got {type(batches)}"

    batch_size = -1
    for batch in batches:
        _validate_basic_batch(batch, train_batch_size, val_batch_size, is_template_batch=False)


def _validate_basic_batch(
    batch: dict[str, torch.Tensor | list[str]],
    train_max_batch_size: int,
    val_max_batch_size: int,
    is_template_batch: bool,
):
    """Function to validate the basics of a batch, either template or data batch. Used to prevent code duplication."""
    ### ASSERT GENERAL STRUCTURE ###
    assert isinstance(batch, dict), f"batch should be a dict, but got {type(batch)}"
    expected_keys = {
        "generation_label",
        "labels",
        "prompt",
        "templates",
        "tokens",
    }
    if is_template_batch:
        expected_keys.add("sample_count")
    assert (
        set(batch.keys()) == expected_keys
    ), f"The 'batch' dictionary should contain {', '.join(expected_keys)}, but contains: {batch.keys()}"
    batch_size = -1

    ### ASSERT INDIVIDUAL LENGTHS ###
    for key, item in batch.items():
        # The templates are list of strings in comparision to the rest which are tensors
        if key == "templates":
            assert all(
                isinstance(template, str) for template in item
            ), f"All templates should be strings, but got {item}"
        else:
            assert isinstance(
                item, torch.Tensor
            ), f"{key} should be tensor, but got {type(item)}"
            if not key == "sample_count":
                if batch_size == -1:
                    batch_size = item.shape[0]
                else:
                    assert (
                        item.shape[0] == batch_size
                    ), f"All tensors in 'batch' should have the same length (except sample_count), but got {len(item)} for key '{key}'"

    if is_template_batch:
        # Since we pack samples into a batch, the batch size can vary, but it should never be bigger than the max_batch_size
        # Since train and val batch size could be different we need to check both (Not perfect)
        assert (
            0 < batch_size <= train_max_batch_size or batch_size <= val_max_batch_size
        ), f"Batch size must be greater than 0 and less than or equal to max_batch_size: for train: {train_max_batch_size} or for val {val_max_batch_size}, but got {batch_size}"
        # The sample count tells you how many examples are in the batch per sample, so the sum of the sample_count should equal the batch size
        assert (
            sum(batch["sample_count"]) == batch_size
        ), f"Sum of sample_count {sum(batch['sample_count'])} must equal batch size {batch_size}"
    else:
        # For data batches we need to meet the batch size exactly
        assert (
            batch_size == train_max_batch_size or batch_size == val_max_batch_size
        ), f"Batch size must equal max_batch_size: for train: {train_max_batch_size} or for val {val_max_batch_size}, but got {batch_size}"

    assert (
        len(batch["templates"]) == batch_size
    ), f"Number of templates {len(batch['templates'])} must equal batch size {batch_size}"

    ### ASSERT INDIVIDUAL CONTENTS ###
    # Check that the generation_label ends with 128009 and then n 128001
    for gen_label in batch["generation_label"]:
        for i, label_id in enumerate(reversed(gen_label)):
            if label_id == 128001:
                continue
            assert (
                label_id == 128009
            ), f"Generation label should end with 128009, but got {label_id} at position {gen_label.shape[0] - i}"
            break

    # Check the prompt starts with n 128001 then 128000, 128006, 882, 128007 and ends with  128009, 128006,  78191, 128007, 271
    for prompt in batch["prompt"]:
        for i, prompt_id in enumerate(prompt):
            if prompt_id == 128001:
                continue
            assert [prompt_id, prompt[i + 1], prompt[i + 2], prompt[i + 3]] == [
                128000,
                128006,
                882,
                128007,
            ], f"Prompt should start with 128001, 128006, 882, 128007, but got {[prompt_id + prompt[i+1] + prompt[i+2]]} at position {i}"
            break
        assert (
            prompt[-5:] == torch.tensor([128009, 128006, 78191, 128007, 271])
        ).all(), f"Prompt should end with [128009, 128006, 78191, 128007, 271], but got {prompt[-5:]}"

    # tokens = prompt + generation_label if we remove all padding tokens
    # Note tokens = prompt + generation_label does not need to hold with padding as we pad tokens once and the prompt and generation_label are padded separately
    # This means sometimes the prompt and generation tokens will need more padding than the tokens

    for prompt, gen_label, tokens in zip(
        batch["prompt"],
        batch["generation_label"],
        batch["tokens"],
    ):
        tokens = tokens[tokens != 128004]
        prompt = prompt[prompt != 128001]
        # Slight hack. We do not include the EOS token (128001) in the generation_label, as we do not want the model to generate it. The dialog could go on.
        # But we use the EOS token as padding for the generation_label. To make it equal to the tokens, we need to ensure that there is exactly one EOS token at the end of the generation_label
        gen_label = gen_label[gen_label != 128001]
        gen_label = torch.cat((gen_label, torch.tensor([128001])), dim=0)
        assert (
            tokens == torch.cat((prompt, gen_label))
        ).all(), f"Tokens should be the concatenation of prompt and generation_label, but got {tokens} != {torch.cat((prompt, gen_label))}"

    # Check that the labels are equivalent to the tokens, except for the 128004 which are -100 for labels and labels are shifted
    labels = batch["labels"]
    labels[labels == -100] = 128004

    labels = torch.cat(
        (torch.ones((labels.shape[0], 1)) * 128000, labels[:, :-1]), dim=1
    )

    # Little bit hacky but we do not calclate the loss on the EOS token (128001) so we have set it to -100. So we need to set this token to 128001
    mask = labels == batch["tokens"]
    assert (~mask).sum() == labels.shape[
        0
    ], "There should be only one token (the EOS token) in each label where token and label differ"
    labels[~mask] = 128001
    assert (
        batch["tokens"] == labels
    ).all(), "Tokens and labels should be equivalent, except for the 128004 which are -100 for labels"
