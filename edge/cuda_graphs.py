"""Optional CUDA-graph acceleration for steady-state bank streaming at fixed resolution."""

from __future__ import annotations

import torch
import torch.nn as nn


class _GraphedFrameBlock(nn.Module):
    def __init__(self, graph: torch.cuda.CUDAGraph, block: nn.Module, static_x, static_pos, static_y):
        super().__init__()
        self.block = block  # keep params alive for the graph
        self.graph = graph
        self.static_x = static_x
        self.static_pos = static_pos
        self.static_y = static_y

    def forward(self, x, pos=None, attn_mask=None, kv_cache=None):
        self.static_x.copy_(x)
        if pos is not None:
            self.static_pos.copy_(pos)
        self.graph.replay()
        return self.static_y.clone()


class _GraphedGlobalBlock(nn.Module):
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        block: nn.Module,
        static_x,
        static_pos,
        static_k,
        static_v,
        static_y,
        static_k_out,
        static_v_out,
    ):
        super().__init__()
        self.block = block
        self.graph = graph
        self.static_x = static_x
        self.static_pos = static_pos
        self.static_k = static_k
        self.static_v = static_v
        self.static_y = static_y
        self.static_k_out = static_k_out
        self.static_v_out = static_v_out

    def forward(self, x, pos=None, attn_mask=None, kv_cache=None):
        if kv_cache is None or kv_cache[0] is None or kv_cache[1] is None:
            # First frame / empty bank — cannot replay the fixed-shape graph.
            return self.block(x, pos, attn_mask, kv_cache)
        if kv_cache[0].shape != self.static_k.shape:
            return self.block(x, pos, attn_mask, kv_cache)
        self.static_x.copy_(x)
        if pos is not None:
            self.static_pos.copy_(pos)
        self.static_k.copy_(kv_cache[0])
        self.static_v.copy_(kv_cache[1])
        self.graph.replay()
        return self.static_y.clone(), [self.static_k_out.clone(), self.static_v_out.clone()]


@torch.inference_mode()
def install_cuda_graphs(
    model: nn.Module,
    height: int,
    width: int,
    window_size: int = 5,
    dtype: torch.dtype = torch.bfloat16,
    sample_image: torch.Tensor | None = None,
) -> None:
    """
    Replace aggregator frame/global blocks with forward-only CUDA graphs.

    Prefer passing ``sample_image`` [1,3,H,W] or [1,1,3,H,W] so graphs are
    captured on real activations (capturing on zeros is numerically unstable).
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    agg = model.aggregator
    if getattr(agg, "_cuda_graphs_installed", False):
        return

    patch = int(agg.patch_size)
    P = (height // patch) * (width // patch) + int(agg.patch_start_idx)
    embed_dim = int(agg.frame_blocks[0].norm1.normalized_shape[0])
    num_heads = int(agg.frame_blocks[0].attn.num_heads)
    head_dim = embed_dim // num_heads
    n_cache = 1 + int(window_size)

    eager_frames = list(agg.frame_blocks)
    eager_globals = list(agg.global_blocks)

    # Build realistic sample activations via a short eager probe when possible.
    sample_tok = None
    sample_pos = None
    sample_k = None
    sample_v = None
    if sample_image is not None:
        img = sample_image
        if img.ndim == 4:
            img = img.unsqueeze(0)
        cache = [[None, None] for _ in range(agg.depth)]
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            for _ in range(n_cache + 1):
                _tokens, _psi, cache = agg(img, mode="window", kv_cache_list=cache)
                for li in range(agg.depth):
                    for kv in range(2):
                        t = cache[li][kv]
                        cache[li][kv] = torch.cat(
                            [t[:, :, :P], t[:, :, max(P, t.size(2) - n_cache * P) :]], dim=2
                        )
            # One more step to get current-frame tokens + prior cache of size n_cache*P
            images = (img - agg._resnet_mean) / agg._resnet_std
            B, S, C_in, H, W = images.shape
            flat = images.view(B * S, C_in, H, W)
            patch_tokens = agg.patch_embed(flat)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            from edge.models.components.aggregator.edge_aggregator import slice_expand_and_flatten

            cam = slice_expand_and_flatten(agg.camera_token, B, S, is_anchor_exist=False)
            reg = slice_expand_and_flatten(agg.register_token, B, S, is_anchor_exist=False)
            tokens = torch.cat([cam, reg, patch_tokens], dim=1)
            pos = agg.position_getter(B * S, H // patch, W // patch, device=images.device)
            pos = pos + 1
            pos_special = torch.zeros(B * S, agg.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
            sample_tok = tokens.detach()
            sample_pos = pos.detach().long()
            sample_k = cache[0][0][:, :, : n_cache * P].detach()
            sample_v = cache[0][1][:, :, : n_cache * P].detach()

    def _frame_bufs():
        if sample_tok is not None:
            sx = sample_tok.clone()
            sp = sample_pos.clone()
        else:
            sx = torch.randn(1, P, embed_dim, device=device, dtype=torch.float32) * 0.02
            sp = torch.zeros(1, P, 2, device=device, dtype=torch.long)
        sy = torch.empty_like(sx)
        return sx, sp, sy

    graphed_frames = []
    for block in eager_frames:
        static_x, static_pos, static_y = _frame_bufs()
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            for _ in range(3):
                static_y.copy_(block(static_x, static_pos))
        g = torch.cuda.CUDAGraph()
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            with torch.cuda.graph(g):
                static_y.copy_(block(static_x, static_pos))
        graphed_frames.append(_GraphedFrameBlock(g, block, static_x, static_pos, static_y))

    graphed_globals = []
    for bi, block in enumerate(eager_globals):
        static_x, static_pos, static_y = _frame_bufs()
        if sample_k is not None:
            # Use matching layer's cache shape from probe when available.
            static_k = cache[bi][0][:, :, : n_cache * P].detach().clone()
            static_v = cache[bi][1][:, :, : n_cache * P].detach().clone()
        else:
            static_k = torch.randn(1, num_heads, n_cache * P, head_dim, device=device, dtype=torch.float32) * 0.02
            static_v = torch.randn(1, num_heads, n_cache * P, head_dim, device=device, dtype=torch.float32) * 0.02
        static_k_out = torch.empty(1, num_heads, (n_cache + 1) * P, head_dim, device=device, dtype=static_k.dtype)
        static_v_out = torch.empty_like(static_k_out)
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            for _ in range(3):
                y, c = block(static_x, static_pos, None, [static_k, static_v])
                static_y.copy_(y)
                static_k_out.copy_(c[0])
                static_v_out.copy_(c[1])
        g = torch.cuda.CUDAGraph()
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            with torch.cuda.graph(g):
                y, c = block(static_x, static_pos, None, [static_k, static_v])
                static_y.copy_(y)
                static_k_out.copy_(c[0])
                static_v_out.copy_(c[1])
        graphed_globals.append(
            _GraphedGlobalBlock(
                g, block, static_x, static_pos, static_k, static_v, static_y, static_k_out, static_v_out
            )
        )

    agg._cuda_graph_keep_alive = {"frame": eager_frames, "global": eager_globals}
    agg.frame_blocks = nn.ModuleList(graphed_frames)
    agg.global_blocks = nn.ModuleList(graphed_globals)
    agg._cuda_graphs_installed = True
