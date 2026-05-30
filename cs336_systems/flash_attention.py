import math
from typing import Any

import torch
from torch.autograd.function import FunctionCtx

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


FlashAttentionFunctionType = type[torch.autograd.Function]


class FlashAttentionPytorchFunction(torch.autograd.Function):
    """Pure PyTorch scaffold for Problem (flash_forward), part (a)."""

    Q_TILE_SIZE: int = 16
    K_TILE_SIZE: int = 16

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Return attention output O and save L, Q, K, V, O for backward."""

        ctx.is_causal = is_causal

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        Q_TILE_SIZE = FlashAttentionPytorchFunction.Q_TILE_SIZE
        K_TILE_SIZE = FlashAttentionPytorchFunction.K_TILE_SIZE
        T_q = math.ceil(n_queries / Q_TILE_SIZE)
        T_k = math.ceil(n_keys / K_TILE_SIZE)

        output = torch.empty_like(Q)  # [B, N, D]
        L = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)  # [B, N]

        for i in range(T_q):
            q_start = i * Q_TILE_SIZE
            q_end = q_start + Q_TILE_SIZE

            Q_i = Q[:, q_start:q_end, :]  # [B, q, D]
            O_i = torch.zeros_like(Q_i)  # [B, q, D]
            m_i_jm1 = torch.full(Q_i.shape[:2], float("-inf"), device=Q.device, dtype=torch.float32)  # [B, q]
            l_i = torch.zeros(Q_i.shape[:2], device=Q.device, dtype=torch.float32)  # [B, q]

            for j in range(T_k):
                k_start = j * K_TILE_SIZE
                k_end = k_start + K_TILE_SIZE

                K_j = K[:, k_start:k_end, :]  # [B, k, D]
                V_j = V[:, k_start:k_end, :]  # [B, k, D]

                S_i_j = Q_i @ K_j.transpose(-2, -1) / math.sqrt(d)  # [B, q, k]
                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=Q.device)  # [q]
                    k_idx = torch.arange(k_start, k_end, device=Q.device)  # [k]
                    causal_mask = q_idx[:, None] >= k_idx[None, :]  # [q, k]
                    S_i_j = torch.where(causal_mask[None, :, :], S_i_j, -1e6)  # [B, q, k]

                m_i_j = torch.maximum(m_i_jm1, S_i_j.max(dim=-1).values)  # [B, q]
                P_i_j = (S_i_j - m_i_j[..., None]).exp()  # [B, q, k]
                alpha = (m_i_jm1 - m_i_j).exp()  # [B, q]
                l_i = alpha * l_i + P_i_j.sum(dim=-1)  # [B, q]
                O_i = alpha[..., None] * O_i + P_i_j @ V_j  # [B, q, D]

                m_i_jm1 = m_i_j  # [B, q]

            O_i = O_i / l_i[..., None]  # [B, q, D]
            L_i = m_i_jm1 + l_i.log()  # [B, q]
            output[:, q_start:q_end, :] = O_i  # [B, q, D]
            L[:, q_start:q_end] = L_i  # [B, q]

        ctx.save_for_backward(L, Q, K, V, output)
        return output

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        """Return gradients for Q, K, V, and no gradient for is_causal."""
        L, Q, K, V, output = ctx.saved_tensors  # [B, N], [B, N, D], [B, M, D], [B, M, D], [B, N, D]
        dO = grad_output  # [B, N, D]

        D = (output * dO).sum(dim=-1)  # [B, N]

        _, n_queries, d = Q.shape
        S = Q @ K.transpose(-2, -1) / math.sqrt(d)  # [B, N, M]
        if ctx.is_causal:
            q_idx = torch.arange(n_queries, device=Q.device)  # [N]
            k_idx = torch.arange(K.shape[-2], device=Q.device)  # [M]
            causal_mask = q_idx[:, None] >= k_idx[None, :]  # [N, M]
            S = torch.where(causal_mask[None, :, :], S, -1e6)  # [B, N, M]

        P = (S - L[..., None]).exp()  # [B, N, M]
        dV = P.transpose(-2, -1) @ dO  # [B, M, D]
        dP = dO @ V.transpose(-2, -1)  # [B, N, M]
        dS = P * (dP - D[..., None])  # [B, N, M]
        dQ = dS @ K / math.sqrt(d)  # [B, N, D]
        dK = dS.transpose(-2, -1) @ Q / math.sqrt(d)  # [B, M, D]
        return dQ, dK, dV, None


if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """Triton scaffold for the FlashAttention-2 forward kernel."""
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # TODO: Implement Algorithm 1:
        # - load Q_block_ptr once for this program
        # - initialize O_i, l_i, and m_i in tl.float32
        # - loop over K/V tiles with K_block_ptr and V_block_ptr
        # - optionally apply the causal mask when is_causal is true
        # - store the final O tile through O_block_ptr and L tile through L_block_ptr
        _ = (Q_block_ptr, K_block_ptr, V_block_ptr, O_block_ptr, L_block_ptr, scale, is_causal)
        raise NotImplementedError("flash_fwd_kernel is scaffolded but not implemented yet")

else:

    def flash_fwd_kernel(*args: Any, **kwargs: Any) -> None:
        """Placeholder used when Triton is not installed in the local environment."""
        raise ModuleNotFoundError("Triton is required to launch flash_fwd_kernel")


class FlashAttentionTritonFunction(torch.autograd.Function):
    """Triton scaffold for Problem (flash_forward), part (b)."""

    Q_TILE_SIZE: int = 16
    K_TILE_SIZE: int = 16

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Launch the Triton FlashAttention-2 forward kernel."""
        if triton is None:
            raise ModuleNotFoundError("Triton is required to run FlashAttentionTritonFunction")

        ctx.is_causal = is_causal

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        Q_TILE_SIZE = FlashAttentionTritonFunction.Q_TILE_SIZE
        K_TILE_SIZE = FlashAttentionTritonFunction.K_TILE_SIZE
        output = torch.empty_like(Q)  # [B, N, D]
        L = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)  # [B, N]
        scale = 1.0 / math.sqrt(d)
        grid = (triton.cdiv(n_queries, Q_TILE_SIZE), batch_size)

        # TODO: Invoke flash_fwd_kernel[grid](...) with Q, K, V, output, L,
        # tensor strides, N_QUERIES, N_KEYS, scale, D, tile sizes, and is_causal.
        _ = (K, V, n_keys, K_TILE_SIZE, scale, grid)
        raise NotImplementedError("FlashAttentionTritonFunction.forward is scaffolded but not implemented yet")

        ctx.save_for_backward(L, Q, K, V, output)
        return output

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        """Compute gradients with respect to Q, K, and V."""
        raise NotImplementedError("FlashAttentionTritonFunction.backward is not implemented yet")


def get_flashattention_autograd_function_pytorch() -> FlashAttentionFunctionType:
    """Return the pure PyTorch FlashAttention-2 autograd.Function class."""
    return FlashAttentionPytorchFunction


def get_flashattention_autograd_function_triton() -> FlashAttentionFunctionType:
    """Return the Triton FlashAttention-2 autograd.Function class."""
    return FlashAttentionTritonFunction
