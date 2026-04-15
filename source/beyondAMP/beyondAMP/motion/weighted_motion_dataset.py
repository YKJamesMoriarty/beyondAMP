from __future__ import annotations

import torch
from typing import List, Literal
from isaaclab.utils import configclass
from .motion_dataset import MotionDataset, MotionDatasetCfg

class WeightedMotionDataset(MotionDataset):
    """
    Extend MotionDataset with weighted sampling on transition pairs (t, t+1).
    """

    def __init__(
        self,
        cfg: MotionDatasetCfg,
        env,
        device="cpu",
        traj_weights: List[float] | None = None,
        transition_weights: torch.Tensor | None = None,
    ):
        super().__init__(cfg, env, device)
        num_histories = len(self.index_hist_end)
        if transition_weights is not None:
            assert transition_weights.shape[0] == num_histories
            self.weights = transition_weights.to(device).clone()
        else:
            self.weights = torch.ones(num_histories).to(device)

        self._traj_weights = traj_weights
        self.norm_weights()

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------
    def norm_weights(self):
        self.weights = self.weights / (self.weights.sum() + 1e-9)

    def update_weights(self, weights: torch.Tensor, method: Literal["sum", "mean", "replace"]="sum", inplace=True):
        if method in ["sum", "mean"]:
            self.weights += weights
            self.norm_weights()
        elif method == "replace":
            self.weights.copy_(weights.to(self.device))
            self.norm_weights()
        else:
            raise NotImplementedError(f"Method: {method} not implemented.")

    # ------------------------------------------------------------------
    # Build from trajectory-level weights
    # ------------------------------------------------------------------
    def _build_transition_weights_from_traj(self, traj_weights):
        """
        Convert trajectory-level weights to transition-level weights.

        Params:
            traj_weights (List[float] or None)

        Returns:
            Tensor of shape (#transitions,)
        """
        if traj_weights is None:
            return torch.ones(len(self.index_hist_end))

        traj_weights = torch.tensor(traj_weights, dtype=torch.float32)

        weights = []
        for w, L in zip(traj_weights, self.traj_lengths):
            if L >= self.history_steps:
                # L frames -> L-H+1 history windows
                weights.append(torch.full((L - self.history_steps + 1,), float(w)))

        return torch.cat(weights, dim=0)

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------
    def sample_batch(self, batch_size: int, replacement = True):
        """兼容旧接口：返回 (t, t+1) transition 索引。"""
        idx = torch.multinomial(self.weights, batch_size, replacement=replacement)
        idx_end = self.index_hist_end[idx]
        t = idx_end - 1
        tp1 = idx_end
        return t, tp1

    def sample_history_batch(self, batch_size: int, replacement=True):
        """按权重采样判别器历史窗口索引 [B, H]。"""
        idx = torch.multinomial(self.weights, batch_size, replacement=replacement)
        idx_end = self.index_hist_end[idx]
        return self._build_history_indices_from_end(idx_end)


@configclass
class WeightedMotionDatasetCfg(MotionDatasetCfg):
    class_type: type[WeightedMotionDataset] = WeightedMotionDataset
