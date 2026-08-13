import torch
from edge.models.edge import Edge


class EdgeSession:
    """
    Streaming inference session (bank mode only).

    Fixed-length GPU memory bank + short working KV cache.
    Retrieval = {anchor} ∪ {recent ``recent_r``} ∪ TopK(bank).
    ``window_size`` controls the GPU working-set size.
    """

    def __init__(
        self,
        model: Edge,
        window_size: int = 5,
        bank_size: int = 32,
        recent_r: int = 2,
        top_k: int = 3,
        depth_only: bool = True,
        pad_working_set: bool = False,
    ):
        self.model = model
        self.window_size = int(window_size)
        self.bank_size = int(bank_size)
        self.recent_r = int(recent_r)
        self.top_k = int(top_k)
        self.depth_only = bool(depth_only)
        self.pad_working_set = bool(pad_working_set)

        self.aggregator_kv_cache_depth = model.aggregator.depth
        self.camera_head_kv_cache_depth = model.camera_head.trunk_depth
        self.camera_head_iterations = 4
        self.patch_start_idx = model.aggregator.patch_start_idx
        self.patch_size = model.aggregator.patch_size

        if self.bank_size < 2:
            raise ValueError("bank_size must be >= 2")

        self.clear()

    def _clear_predictions(self):
        self.predictions = dict()

    def _update_predictions(self, predictions):
        # Keep only the latest frame to minimize host/GPU memory traffic.
        keep = ("depth", "depth_conf", "images") if self.depth_only else (
            "pose_enc",
            "world_points",
            "world_points_conf",
            "depth",
            "depth_conf",
            "images",
        )
        for k in keep:
            if k in predictions:
                self.predictions[k] = predictions[k]

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
        self.camera_head_kv_cache_list = None if self.depth_only else [
            [[None, None] for _ in range(self.camera_head_kv_cache_depth)]
            for _ in range(self.camera_head_iterations)
        ]
        # GPU-resident bank (H100 has enough VRAM; avoids CPU↔GPU ping-pong).
        self.agg_bank = [[[], []] for _ in range(self.aggregator_kv_cache_depth)]
        self.cam_bank = None if self.depth_only else [
            [[[], []] for _ in range(self.camera_head_kv_cache_depth)]
            for _ in range(self.camera_head_iterations)
        ]
        self.bank_summaries = []
        self.num_bank_frames = 0
        self._query_summary = None
        self._tokens_per_frame_cached = None

    def _tokens_per_frame(self) -> int:
        if self._tokens_per_frame_cached is not None:
            return self._tokens_per_frame_cached
        h = int(self.predictions["depth"].shape[2])
        w = int(self.predictions["depth"].shape[3])
        self._tokens_per_frame_cached = (
            h * w // self.patch_size // self.patch_size + self.patch_start_idx
        )
        return self._tokens_per_frame_cached

    def _model_device(self):
        return next(self.model.parameters()).device

    def _score_bank_frames(self) -> torch.Tensor:
        n = self.num_bank_frames
        if n == 0:
            return torch.zeros(0, device=self._model_device())
        if self._query_summary is None or len(self.bank_summaries) == 0:
            return torch.arange(n, dtype=torch.float32, device=self._model_device())

        q = self._query_summary.float()
        keys = torch.stack(self.bank_summaries, dim=0).float()
        q_n = torch.nn.functional.normalize(q, dim=-1)
        k_n = torch.nn.functional.normalize(keys, dim=-1)
        return (k_n * q_n.unsqueeze(0)).sum(dim=-1).mean(dim=(1, 2))

    def _select_working_indices(self) -> list[int]:
        n = self.num_bank_frames
        if n == 0:
            return []
        max_working = 1 + self.window_size
        if n <= max_working:
            return list(range(n))

        forced = {0}
        for r in range(1, min(self.recent_r, n - 1) + 1):
            forced.add(n - r)

        scores = self._score_bank_frames()
        cand = [i for i in range(n) if i not in forced]
        # TopK on GPU then move indices only.
        if cand:
            cand_t = torch.tensor(cand, device=scores.device, dtype=torch.long)
            cand_scores = scores[cand_t]
            k = min(self.top_k, len(cand))
            top = torch.topk(cand_scores, k=k, largest=True).indices
            chosen = cand_t[top].tolist()
        else:
            chosen = []

        selected = set(forced)
        for i in chosen:
            selected.add(int(i))
            if len(selected) >= max_working:
                break
        if len(selected) < max_working:
            for i in sorted(cand, key=lambda j: float(scores[j]), reverse=True):
                if i in selected:
                    continue
                selected.add(i)
                if len(selected) >= max_working:
                    break
        return sorted(selected)

    def _gather_working_cache(self):
        idxs = self._select_working_indices()
        n_target = 1 + self.window_size

        if not idxs:
            self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
            if not self.depth_only:
                self.camera_head_kv_cache_list = [
                    [[None, None] for _ in range(self.camera_head_kv_cache_depth)]
                    for _ in range(self.camera_head_iterations)
                ]
            return

        # Pad only when requested (CUDA graphs need fixed working-set length).
        if self.pad_working_set and len(idxs) < n_target:
            idxs = list(idxs) + [idxs[-1]] * (n_target - len(idxs))

        for layer in range(self.aggregator_kv_cache_depth):
            for kv in range(2):
                chunks = self.agg_bank[layer][kv]
                self.aggregator_kv_cache_list[layer][kv] = torch.cat(
                    [chunks[i] for i in idxs], dim=2
                )

        if not self.depth_only:
            for it in range(self.camera_head_iterations):
                for layer in range(self.camera_head_kv_cache_depth):
                    for kv in range(2):
                        self.camera_head_kv_cache_list[it][layer][kv] = torch.cat(
                            [self.cam_bank[it][layer][kv][i] for i in idxs], dim=2
                        )

    def _append_current_to_bank(self, aggregator_kv_cache_list, camera_head_kv_cache_list):
        P = self._tokens_per_frame()

        for layer in range(self.aggregator_kv_cache_depth):
            for kv in range(2):
                # contiguous() copies when needed; avoid extra clone
                cur = aggregator_kv_cache_list[layer][kv][:, :, -P:].detach().contiguous()
                self.agg_bank[layer][kv].append(cur)

        if not self.depth_only and camera_head_kv_cache_list is not None:
            for it in range(self.camera_head_iterations):
                for layer in range(self.camera_head_kv_cache_depth):
                    for kv in range(2):
                        cur = camera_head_kv_cache_list[it][layer][kv][:, :, -1:].detach().contiguous()
                        self.cam_bank[it][layer][kv].append(cur)

        cam_ks = [
            aggregator_kv_cache_list[layer][0][:, :, -P : -P + 1, :]
            for layer in range(self.aggregator_kv_cache_depth)
        ]
        summary = torch.stack(cam_ks, dim=0).mean(dim=0).squeeze(2).detach().contiguous()
        self.bank_summaries.append(summary)
        self._query_summary = summary
        self.num_bank_frames += 1
        self._evict_bank_if_needed()

    def _evict_bank_if_needed(self):
        while self.num_bank_frames > self.bank_size:
            n = self.num_bank_frames
            protected = {0}
            for r in range(1, self.recent_r + 1):
                protected.add(n - r)
            evict = None
            for i in range(1, n - self.recent_r):
                if i not in protected:
                    evict = i
                    break
            if evict is None:
                evict = 1 if n > 1 else 0
            self._drop_bank_index(evict)

    def _drop_bank_index(self, idx: int):
        for layer in range(self.aggregator_kv_cache_depth):
            for kv in range(2):
                del self.agg_bank[layer][kv][idx]
        if not self.depth_only:
            for it in range(self.camera_head_iterations):
                for layer in range(self.camera_head_kv_cache_depth):
                    for kv in range(2):
                        del self.cam_bank[it][layer][kv][idx]
        del self.bank_summaries[idx]
        self.num_bank_frames -= 1

    def get_all_predictions(self):
        return self.predictions

    def get_last_prediction(self):
        return self.predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()

    def forward_stream(self, images):
        self._gather_working_cache()

        outputs = self.model(
            images=images,
            mode="window",
            aggregator_kv_cache_list=self.aggregator_kv_cache_list,
            camera_head_kv_cache_list=None if self.depth_only else self.camera_head_kv_cache_list,
            depth_only=self.depth_only,
        )

        self._update_predictions(outputs)
        self._append_current_to_bank(
            outputs["aggregator_kv_cache_list"],
            outputs.get("camera_head_kv_cache_list"),
        )
        # Drop working cache; next step re-gathers from GPU bank.
        self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
        if not self.depth_only:
            self.camera_head_kv_cache_list = [
                [[None, None] for _ in range(self.camera_head_kv_cache_depth)]
                for _ in range(self.camera_head_iterations)
            ]

        return self.get_all_predictions()
