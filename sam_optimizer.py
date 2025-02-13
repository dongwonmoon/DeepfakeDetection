import torch
from torch.optim.optimizer import Optimizer


class SAM(Optimizer):
    """
    Implements Sharpness-Aware Minimization (SAM).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        base_optimizer (torch.optim.Optimizer): underlying optimizer (e.g., torch.optim.Adam).
        rho (float): size of the neighborhood for SAM.
        adaptive (bool): if True, adapt the perturbation according to the parameter norm.
        **kwargs: additional arguments for the base optimizer.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError("Invalid rho value: should be non-negative")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer(params, **kwargs)
        super(SAM, self).__init__(params, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # Compute the norm of gradients.
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # For adaptive SAM the perturbation is scaled by the parameter norm.
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)  # climb to the local maximum
                p.state["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # revert to original parameters and take an optimizer step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in p.state:
                    continue
                p.sub_(p.state["e_w"])  # revert to original parameter
                p.state.pop("e_w", None)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure):
        """
        Performs a single SAM optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            raise RuntimeError("SAM requires a closure, but none was provided")
        # First forward-backward pass
        loss = closure()
        loss.backward()
        self.first_step(zero_grad=True)
        # Second forward-backward pass
        loss = closure()
        loss.backward()
        self.second_step(zero_grad=True)
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
