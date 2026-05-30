from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class FSDP(torch.nn.Module):
    """Scaffold for fully sharded data parallel training."""

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FSDP requires an initialized torch.distributed process group")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._handles: list[dist.Work] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._sharded_param_metadata: dict[str, dict[str, Any]] = {}

        self._shard_parameters()
        self._register_fsdp_hooks()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Forward all inputs to the wrapped module."""
        return self.module(*inputs, **kwargs)

    def _should_shard_module(self, module: torch.nn.Module) -> bool:
        """Return whether this module's weight should be sharded by FSDP."""
        from cs336_basics.model import Embedding, Linear

        return isinstance(module, (Embedding, Linear)) and isinstance(getattr(module, "weight", None), torch.nn.Parameter)

    def _shard_parameters(self) -> None:
        """Replace selected full parameters with this rank's local shards."""
        with torch.no_grad():
            for module_name, child_module in self.module.named_modules():
                for buffer in child_module.buffers(recurse=False):
                    dist.broadcast(buffer, src=0)

                if not self._should_shard_module(child_module):
                    for param in child_module.parameters(recurse=False):
                        dist.broadcast(param, src=0)
                    continue

                full_weight = child_module.weight
                dist.broadcast(full_weight, src=0)

                full_shape = tuple(full_weight.shape)
                padded_rows = ((full_shape[0] + self.world_size - 1) // self.world_size) * self.world_size
                rows_per_rank = padded_rows // self.world_size
                pad_rows = padded_rows - full_shape[0]

                padded_weight = full_weight.detach()
                if pad_rows:
                    padding = torch.zeros(
                        (pad_rows, *full_shape[1:]),
                        device=full_weight.device,
                        dtype=full_weight.dtype,
                    )
                    padded_weight = torch.cat([padded_weight, padding], dim=0)

                start = self.rank * rows_per_rank
                end = start + rows_per_rank
                local_shard = padded_weight[start:end].contiguous()
                child_module.weight = torch.nn.Parameter(local_shard, requires_grad=full_weight.requires_grad)

                param_name = f"{module_name}.weight" if module_name else "weight"
                self._sharded_param_metadata[param_name] = {
                    "module": child_module,
                    "local_shard_parameter": child_module.weight,
                    "full_shape": full_shape,
                    "padded_shape": tuple(padded_weight.shape),
                    "rows_per_rank": rows_per_rank,
                    "pad_rows": pad_rows,
                }

    def _register_fsdp_hooks(self) -> None:
        """Register hooks for all-gathering weights and reduce-scattering gradients."""
        for name, metadata in self._sharded_param_metadata.items():
            module = metadata["module"]

            def gather_before_forward(module: torch.nn.Module, _inputs: tuple[Any, ...], name: str = name) -> None:
                metadata = self._sharded_param_metadata[name]
                local_shard_parameter = metadata["local_shard_parameter"]
                full_weight = self._all_gather_parameter(name, local_shard_parameter)
                gathered_parameter = torch.nn.Parameter(
                    full_weight,
                    requires_grad=local_shard_parameter.requires_grad,
                )
                metadata["gathered_parameter"] = gathered_parameter
                module.weight = gathered_parameter

                if gathered_parameter.requires_grad:
                    grad_handle = gathered_parameter.register_post_accumulate_grad_hook(lambda parameter, name=name: self._reduce_scatter_gradient(name, parameter))
                    self._hooks.append(grad_handle)

            def free_after_forward(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                _output: Any,
                name: str = name,
            ) -> None:
                self._free_gathered_parameter(name)

            self._hooks.append(module.register_forward_pre_hook(gather_before_forward))
            self._hooks.append(module.register_forward_hook(free_after_forward))

    def _all_gather_parameter(self, name: str, parameter: torch.nn.Parameter, cast_to_compute_dtype=True) -> torch.Tensor:
        """All-gather one sharded parameter into its full tensor form."""
        metadata = self._sharded_param_metadata[name]
        gathered = [torch.empty_like(parameter.data) for _ in range(self.world_size)]
        dist.all_gather(gathered, parameter.data)

        full_parameter = torch.cat(gathered, dim=0)
        full_shape = metadata["full_shape"]
        full_parameter = full_parameter[: full_shape[0]].view(full_shape).contiguous()
        if cast_to_compute_dtype and self.compute_dtype is not None:
            full_parameter = full_parameter.to(self.compute_dtype)
        metadata["gathered_parameter"] = full_parameter
        return full_parameter

    def _free_gathered_parameter(self, name: str) -> None:
        """Free a previously all-gathered parameter after use."""
        metadata = self._sharded_param_metadata[name]
        module = metadata["module"]
        module.weight = metadata["local_shard_parameter"]
        metadata.pop("gathered_parameter", None)

    def _reduce_scatter_gradient(self, name: str, parameter: torch.nn.Parameter) -> None:
        """Reduce-scatter a full gradient into this rank's gradient shard."""
        metadata = self._sharded_param_metadata[name]
        local_shard_parameter = metadata["local_shard_parameter"]

        full_grad = parameter.grad
        if full_grad is None:
            return

        full_grad = full_grad.to(local_shard_parameter.dtype)
        pad_rows = metadata["padded_shape"][0] - full_grad.shape[0]
        if pad_rows:
            padding = torch.zeros(
                (pad_rows, *full_grad.shape[1:]),
                device=full_grad.device,
                dtype=full_grad.dtype,
            )
            padded_grad = torch.cat([full_grad, padding], dim=0)
        else:
            padded_grad = full_grad

        padded_grad = padded_grad / self.world_size
        if dist.get_backend() == "gloo":
            dist.all_reduce(padded_grad, op=dist.ReduceOp.SUM)
            local_grad = padded_grad.chunk(self.world_size, dim=0)[self.rank].contiguous()
        else:
            grad_chunks = padded_grad.chunk(self.world_size, dim=0)
            local_grad = torch.empty_like(local_shard_parameter.data)
            dist.reduce_scatter(local_grad, list(grad_chunks), op=dist.ReduceOp.SUM)

        local_shard_parameter.grad = local_grad
        parameter.grad = None

    def finish_gradient_synchronization(self) -> None:
        """Wait for outstanding FSDP communication work."""
        for handle in self._handles:
            handle.wait()

        self._handles.clear()

        for name in self._sharded_param_metadata:
            self._free_gathered_parameter(name)

        for name, param in self.module.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            if name in self._sharded_param_metadata:
                continue

            param.grad /= self.world_size
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Return a full state dict reconstructed from this rank's sharded parameters."""
        full_state = {}
        for name, param in self.module.named_parameters():
            if name in self._sharded_param_metadata:
                full_param = self._all_gather_parameter(name, param, cast_to_compute_dtype=False)
                full_state[name] = full_param.detach().clone()
                self._free_gathered_parameter(name)
            else:
                full_state[name] = param.detach().clone()
        return full_state


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """Return an FSDP wrapper around module."""
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Run FSDP post-backward work before optimizer.step()."""
    del optimizer
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Gather full parameters from an FSDP model."""
    return fsdp_model.gather_full_params()
