import torch

class Optimizer:
    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass

class Model:
    def parameters(self):
        return []


def _loss_step(template_batch):
    return 0.0

def _optimizer_backward(loss):
    pass

epoch = []
gradient_accumulation_steps = 1
optimizer = Optimizer()
model = Model()

def get_gradients(per_param_lists, params):
    for i,p in enumerate(params):
        if p.grad is not None:
            per_param_lists[i].append(p.grad.detach().clone())
    return per_param_lists


def aggregate(per_param_lists, grads):
    "This function aggregates gradients for all template batches and adds it to the grads list."
    for i, grad_list in enumerate(per_param_lists):
        if grad_list:
            agg = torch.stack(grad_list).mean(dim=0) # Exchange here for other aggregation methods if needed
            grads[i].add_(agg)
    return grads

def loop():
    params = [p for p in model.parameters() if p.requires_grad] # All model parameters that require gradients
    grads = [torch.zeros_like(p) for p in params] # dataset_batch gradients

    for dataset_batch in epoch:
        per_param_lists = [[] for _ in params] # template_batch gradients
        for template_batch in dataset_batch:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_step(template_batch)
            _optimizer_backward(loss)
            per_param_lists = get_gradients(per_param_lists, params)
        grads = aggregate(per_param_lists, grads)

        if gradient_accumulation_steps:
            # Reassign grads
            with torch.no_grad():
                for p, g in zip(params, grads):
                    p.grad = None if g is None else g.to(p.device)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            grads = [torch.zeros_like(p) for p in params]