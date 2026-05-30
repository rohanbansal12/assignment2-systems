from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim.optimizer import ParamsT


class ShardedOptimizer(torch.optim.Optimizer):
    """Optimizer wrapper that shards optimizer state across distributed ranks."""

    def __init__(
        self,
        params: ParamsT,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires an initialized torch.distributed process group")

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_optimizer: torch.optim.Optimizer | None = None
        self._ready_to_update_local_optimizer = False

        super().__init__(params, defaults={})
        self.local_optimizer = self._build_local_optimizer()
        self._ready_to_update_local_optimizer = True

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a parameter group and assign its parameters across ranks."""
        super().add_param_group(param_group)
        if not self._ready_to_update_local_optimizer:
            return

        if self.local_optimizer is None:
            self.local_optimizer = self._build_local_optimizer()
            return

        local_group = self._local_param_group(self.param_groups[-1])
        if local_group["params"]:
            self.local_optimizer.add_param_group(local_group)

    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any) -> Any:
        """Step the local optimizer, then synchronize updated parameters."""
        result = None
        if self.local_optimizer is not None:
            if closure is None:
                result = self.local_optimizer.step(**kwargs)
            else:
                result = self.local_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                result = closure()

        self._sync_parameters()
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for all parameters managed by this optimizer wrapper."""
        super().zero_grad(set_to_none=set_to_none)

    def _build_local_optimizer(self) -> torch.optim.Optimizer | None:
        """Construct the wrapped optimizer over this rank's parameter shard."""
        local_param_groups = self._local_param_groups()
        if not local_param_groups:
            return None
        return self.optimizer_cls(local_param_groups, **self.optimizer_kwargs)

    def _local_param_groups(self) -> list[dict[str, Any]]:
        """Return parameter groups containing only this rank's assigned parameters."""
        return [local_group for group in self.param_groups if (local_group := self._local_param_group(group))["params"]]

    def _local_param_group(self, param_group: dict[str, Any]) -> dict[str, Any]:
        """Return one parameter group narrowed to this rank's owned parameters."""
        param_to_index = self._parameter_indices()
        local_group = {key: value for key, value in param_group.items() if key != "params"}
        local_group["params"] = [param for param in param_group["params"] if self._parameter_owner(param_to_index[id(param)]) == self.rank]
        return local_group

    def _all_parameters(self) -> list[torch.Tensor]:
        """Return all unique parameters in wrapper order."""
        parameters: list[torch.Tensor] = []
        seen: set[int] = set()
        for group in self.param_groups:
            for param in group["params"]:
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                parameters.append(param)
        return parameters

    def _parameter_indices(self) -> dict[int, int]:
        """Return a stable id-to-index map over all unique parameters."""
        return {id(param): index for index, param in enumerate(self._all_parameters())}

    def _parameter_owner(self, parameter_index: int) -> int:
        """Return the rank responsible for optimizer state and updates for one parameter."""
        return parameter_index % self.world_size

    def _sync_parameters(self) -> None:
        """Broadcast each updated parameter from its owner rank to all other ranks."""
        with torch.no_grad():
            for parameter_index, param in enumerate(self._all_parameters()):
                dist.broadcast(param, src=self._parameter_owner(parameter_index))


def get_sharded_optimizer(
    params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    optimizer_cls: type[torch.optim.Optimizer],
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Return a sharded optimizer wrapper."""
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
