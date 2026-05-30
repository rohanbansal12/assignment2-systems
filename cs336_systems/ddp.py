from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class DDP(torch.nn.Module):
    """Scaffold for an overlapping distributed data parallel wrapper."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self._handles: list[dist.Work] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DDP requires an initialized torch.distributed process group")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._broadcast_parameters()
        self._register_gradient_hooks()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Forward all inputs to the wrapped module."""
        return self.module(*inputs, **kwargs)

    def _broadcast_parameters(self) -> None:
        """Broadcast rank-0 parameters and buffers to all ranks."""
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

    def _register_gradient_hooks(self) -> None:
        """Register post-accumulation hooks for async gradient all-reduce."""
        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            handle = param.register_post_accumulate_grad_hook(
                self._make_gradient_hook(param)
            )
            self._hooks.append(handle)

    def _make_gradient_hook(self, parameter: torch.nn.Parameter):
        """Create a hook that starts async gradient averaging for one parameter."""
        def hook(_parameter: torch.nn.Parameter) -> None:
            if parameter.grad is None:
                return

            parameter.grad /= self.world_size
            handle = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

        return hook

    def finish_gradient_synchronization(self) -> None:
        """Wait for all outstanding async gradient synchronization work."""
        for handle in self._handles:
            handle.wait()
        self._handles.clear()


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """Return a DDP wrapper around module."""
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Run DDP post-backward work before optimizer.step()."""
    del optimizer
    ddp_model.finish_gradient_synchronization()
