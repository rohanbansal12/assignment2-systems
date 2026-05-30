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

        q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_i_jm1 = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k_i_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_i_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            s_i_j = tl.dot(q_i, tl.trans(k_i_j)) * scale
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s_i_j = tl.where(k_offsets[None, :] < N_KEYS, s_i_j, -1e6)
            if is_causal:
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                s_i_j = tl.where(causal_mask, s_i_j, -1e6)
            m_i_j = tl.maximum(m_i_jm1, tl.max(s_i_j, axis=-1))
            p_i_j = tl.exp(s_i_j - m_i_j[:, None])
            alpha = tl.exp(m_i_jm1 - m_i_j)
            l_i = alpha * l_i + tl.sum(p_i_j, axis=-1)
            O_i = alpha[:, None] * O_i
            O_i = tl.dot(p_i_j.to(v_i_j.dtype), v_i_j, acc=O_i)

            m_i_jm1 = m_i_j
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        O_i = O_i / l_i[:, None]
        L_i = m_i_jm1 + tl.log(l_i)

        tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))

    @triton.jit
    def flash_bwd_delta_kernel(
        O_ptr,
        dO_ptr,
        Delta_ptr,
        stride_ob,
        stride_oq,
        stride_od,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_deltab,
        stride_deltaq,
        N_QUERIES,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        Delta_block_ptr = tl.make_block_ptr(
            Delta_ptr + batch_index * stride_deltab,
            shape=(N_QUERIES,),
            strides=(stride_deltaq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Delta_i = tl.sum(O_i * dO_i, axis=-1)

        tl.store(Delta_block_ptr, Delta_i, boundary_check=(0,))

    @triton.jit
    def flash_bwd_dkdv_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        Delta_ptr,
        dK_ptr,
        dV_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_lb,
        stride_lq,
        stride_deltab,
        stride_deltaq,
        stride_dkb,
        stride_dkk,
        stride_dkd,
        stride_dvb,
        stride_dvk,
        stride_dvd,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        key_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(0, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(0, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(0,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        Delta_block_ptr = tl.make_block_ptr(
            Delta_ptr + batch_index * stride_deltab,
            shape=(N_QUERIES,),
            strides=(stride_deltaq,),
            offsets=(0,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        dK_block_ptr = tl.make_block_ptr(
            dK_ptr + batch_index * stride_dkb,
            shape=(N_KEYS, D),
            strides=(stride_dkk, stride_dkd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        dV_block_ptr = tl.make_block_ptr(
            dV_ptr + batch_index * stride_dvb,
            shape=(N_KEYS, D),
            strides=(stride_dvk, stride_dvd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
            q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
            L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            Delta_i = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero")

            q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            S_i_j = tl.dot(q_i, tl.trans(k_j)) * scale
            valid_mask = (q_offsets[:, None] < N_QUERIES) & (k_offsets[None, :] < N_KEYS)
            S_i_j = tl.where(valid_mask, S_i_j, -1e6)
            if is_causal:
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                S_i_j = tl.where(causal_mask, S_i_j, -1e6)

            P_i_j = tl.exp(S_i_j - L_i[:, None])
            dV_j = tl.dot(tl.trans(P_i_j.to(dO_i.dtype)), dO_i, acc=dV_j)
            dP_i_j = tl.dot(dO_i, tl.trans(v_j))
            dS_i_j = P_i_j * (dP_i_j - Delta_i[:, None])
            dK_j = tl.dot(tl.trans(dS_i_j.to(q_i.dtype)), q_i, acc=dK_j)

            Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
            dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
            L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
            Delta_block_ptr = tl.advance(Delta_block_ptr, (Q_TILE_SIZE,))

        tl.store(dK_block_ptr, (dK_j * scale).to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))

    @triton.jit
    def flash_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        Delta_ptr,
        dQ_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_lb,
        stride_lq,
        stride_deltab,
        stride_deltaq,
        stride_dqb,
        stride_dqq,
        stride_dqd,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
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
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
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
        Delta_block_ptr = tl.make_block_ptr(
            Delta_ptr + batch_index * stride_deltab,
            shape=(N_QUERIES,),
            strides=(stride_deltaq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
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
        dQ_block_ptr = tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D),
            strides=(stride_dqq, stride_dqd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        Delta_i = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero")
        dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            S_i_j = tl.dot(q_i, tl.trans(k_j)) * scale
            valid_mask = (q_offsets[:, None] < N_QUERIES) & (k_offsets[None, :] < N_KEYS)
            S_i_j = tl.where(valid_mask, S_i_j, -1e6)
            if is_causal:
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                S_i_j = tl.where(causal_mask, S_i_j, -1e6)

            P_i_j = tl.exp(S_i_j - L_i[:, None])
            dP_i_j = tl.dot(dO_i, tl.trans(v_j))
            dS_i_j = P_i_j * (dP_i_j - Delta_i[:, None])
            dQ_i = tl.dot(dS_i_j.to(k_j.dtype), k_j, acc=dQ_i)

            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        tl.store(dQ_block_ptr, (dQ_i * scale).to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

else:

    def flash_fwd_kernel(*args: Any, **kwargs: Any) -> None:
        """Placeholder used when Triton is not installed in the local environment."""
        raise ModuleNotFoundError("Triton is required to launch flash_fwd_kernel")

    def flash_bwd_delta_kernel(*args: Any, **kwargs: Any) -> None:
        """Placeholder used when Triton is not installed in the local environment."""
        raise ModuleNotFoundError("Triton is required to launch flash_bwd_delta_kernel")

    def flash_bwd_dkdv_kernel(*args: Any, **kwargs: Any) -> None:
        """Placeholder used when Triton is not installed in the local environment."""
        raise ModuleNotFoundError("Triton is required to launch flash_bwd_dkdv_kernel")

    def flash_bwd_dq_kernel(*args: Any, **kwargs: Any) -> None:
        """Placeholder used when Triton is not installed in the local environment."""
        raise ModuleNotFoundError("Triton is required to launch flash_bwd_dq_kernel")


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

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            output,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            L.stride(0),
            L.stride(1),
            n_queries,
            n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, output)
        return output

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        """Compute gradients with respect to Q, K, and V."""
        if triton is None:
            raise ModuleNotFoundError("Triton is required to run FlashAttentionTritonFunction")

        L, Q, K, V, output = ctx.saved_tensors
        dO = grad_output

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        Q_TILE_SIZE = FlashAttentionTritonFunction.Q_TILE_SIZE
        K_TILE_SIZE = FlashAttentionTritonFunction.K_TILE_SIZE
        scale = 1.0 / math.sqrt(d)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        Delta = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

        query_grid = (triton.cdiv(n_queries, Q_TILE_SIZE), batch_size)
        key_grid = (triton.cdiv(n_keys, K_TILE_SIZE), batch_size)

        flash_bwd_delta_kernel[query_grid](
            output,
            dO,
            Delta,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            Delta.stride(0),
            Delta.stride(1),
            n_queries,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
        )

        flash_bwd_dkdv_kernel[key_grid](
            Q,
            K,
            V,
            dO,
            L,
            Delta,
            dK,
            dV,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            L.stride(0),
            L.stride(1),
            Delta.stride(0),
            Delta.stride(1),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            n_queries,
            n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )

        flash_bwd_dq_kernel[query_grid](
            Q,
            K,
            V,
            dO,
            L,
            Delta,
            dQ,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            L.stride(0),
            L.stride(1),
            Delta.stride(0),
            Delta.stride(1),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            n_queries,
            n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )

        return dQ, dK, dV, None
