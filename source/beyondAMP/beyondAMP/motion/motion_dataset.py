from __future__ import annotations

import os
import numpy as np
import torch
from typing import Sequence, List, Union
from dataclasses import MISSING

from isaaclab.utils import configclass

from .utils.math import quat_apply_inverse, quat_conjugate, quat_apply
from .motion_transition import MotionTransition

class MotionDataset:
    """
    Load multiple motion files and build discriminator samples.

    Notes:
    - 保留 (s_t, s_{t+1}) 索引接口，兼容 reset 等已有逻辑。
    - 判别器训练样本改为可配置历史窗口（history_steps 帧）拼接向量。
    """

    def __init__(
        self, 
        cfg: MotionDatasetCfg,
        env,
        device: str = "cpu",
        ):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.robot = env.scene[cfg.asset_name]
        self.motion_files = cfg.motion_files
        self.observation_terms = cfg.amp_obs_terms
        # 判别器历史窗口长度，默认 2（与旧版 transition 输入等价）。
        self.history_steps = max(int(getattr(cfg, "history_steps", 2)), 2)
        
        body_names = cfg.body_names
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(body_names, preserve_order=True)[0], dtype=torch.long, device=device
        )

        anchor_name = cfg.anchor_name
        self.anchor_index = torch.tensor(
            self.robot.find_bodies(anchor_name, preserve_order=True)[0], dtype=torch.long, device=device
        )
        
        self.load_motions()
        self._build_joint_reference()
        self.init_observation_dims()

    def load_motions(self):
        # Storage lists (later concatenated)
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        fps_list = []
        traj_lengths = []

        # Load all motion files
        for f in self.motion_files:
            assert os.path.isfile(f), f"Invalid motion file: {f}"
            data = np.load(f)

            fps_list.append(float(data["fps"]))
            traj_len = data["joint_pos"].shape[0]
            traj_lengths.append(traj_len)

            joint_pos_list.append(torch.tensor(data["joint_pos"], dtype=torch.float32))
            joint_vel_list.append(torch.tensor(data["joint_vel"], dtype=torch.float32))
            body_pos_w_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32))
            body_quat_w_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32))
            body_lin_vel_w_list.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32))
            body_ang_vel_w_list.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32))

        # Concatenate all trajectories into single big tensors.
        # 命名为 *_raw，强调这里是 npz 原始绝对关节量（尚未做相对化）。
        self.joint_pos_raw      = torch.cat(joint_pos_list, dim=0).to(self.device)
        self.joint_vel_raw      = torch.cat(joint_vel_list, dim=0).to(self.device)
        self.body_pos_w_all      = torch.cat(body_pos_w_list, dim=0).to(self.device)
        self.body_quat_w_all     = torch.cat(body_quat_w_list, dim=0).to(self.device)
        self.body_lin_vel_w_all  = torch.cat(body_lin_vel_w_list, dim=0).to(self.device)
        self.body_ang_vel_w_all  = torch.cat(body_ang_vel_w_list, dim=0).to(self.device)

        self.total_dataset_size = sum(traj_lengths)
        self.traj_lengths = traj_lengths

        # Keep per-trajectory FPS if needed
        self.fps_list = fps_list

        # Build transition index list: (global_index_t, global_index_t+1)
        self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths, self.device)
        # Build history-end indices for discriminator window sampling.
        self.index_hist_end = self._build_history_end_indices(traj_lengths, self.history_steps, self.device)

    def _build_joint_reference(self):
        """构建 demo 关节相对化所需的默认关节参考。

        目标：
        1) 让 demo 侧 joint_pos/joint_vel 与 agent 侧 joint_pos_rel/joint_vel_rel 语义一致；
        2) 同时保留 npz 的绝对值接口，供 reset_to_ref_motion_dataset 使用。
        """
        # IsaacLab 默认关节值按 [num_envs, num_joints] 存储，这里取第 0 个环境即可。
        default_joint_pos = self.robot.data.default_joint_pos[0].to(self.device)
        default_joint_vel = self.robot.data.default_joint_vel[0].to(self.device)

        demo_joint_dim = self.joint_pos_raw.shape[-1]
        if default_joint_pos.shape[-1] != demo_joint_dim:
            raise ValueError(
                "Demo joint dimension does not match robot default joint dimension: "
                f"demo={demo_joint_dim}, robot_default={default_joint_pos.shape[-1]}."
            )
        if default_joint_vel.shape[-1] != demo_joint_dim:
            raise ValueError(
                "Demo joint-velocity dimension does not match robot default joint dimension: "
                f"demo={demo_joint_dim}, robot_default={default_joint_vel.shape[-1]}."
            )

        # 形状扩展为 [1, J]，便于与 [N, J] 广播相减。
        self.default_joint_pos_ref = default_joint_pos.unsqueeze(0)
        self.default_joint_vel_ref = default_joint_vel.unsqueeze(0)

    # ----------------------- Property API -----------------------
    
    def subtract_flaten(self, target: torch.Tensor):
        target = target[:, self.body_indexes]
        return target.reshape(self.total_dataset_size, -1)

    @property
    def joint_pos_abs(self):
        """Demo 侧绝对关节角（原始 npz 值）。"""
        return self.joint_pos_raw

    @property
    def joint_vel_abs(self):
        """Demo 侧绝对关节角速度（原始 npz 值）。"""
        return self.joint_vel_raw

    @property
    def joint_pos(self):
        """Demo 侧相对关节角：q_demo_rel = q_demo_abs - q_default。

        说明：
        - 这里与 agent 的 joint_pos_rel 保持一致，避免判别器输入语义不一致。
        - 若需要原始绝对值（例如 reset），请使用 joint_pos_abs。
        """
        return self.joint_pos_raw - self.default_joint_pos_ref

    @property
    def joint_vel(self):
        """Demo 侧相对关节角速度：qd_demo_rel = qd_demo_abs - qd_default。"""
        return self.joint_vel_raw - self.default_joint_vel_ref
    
    @property
    def body_pos_w(self):
        return self.body_pos_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_quat_w(self):
        return self.body_quat_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    
    @property
    def body_pos_b(self):
        """
        body positions expressed in anchor-local frame.
        Output: (N, num_bodies * 3)
        """
        # (N, B, 3)
        pos_w = self.body_pos_w_all[:, self.body_indexes]  

        # (N, 1, 3)
        anchor_pos = self._anchor_pos.unsqueeze(1)
        anchor_quat = self._anchor_quat.unsqueeze(1)

        # translate then rotate into anchor frame
        rel = pos_w - anchor_pos                           # world-space relative
        rel_local = quat_apply_inverse(anchor_quat, rel)   # world → anchor

        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def body_quat_b(self):
        """
        body orientations expressed in anchor-local frame.
        q_local = q_anchor^{-1} ⊗ q_body
        Output: (N, num_bodies * 4)
        """
        q_body = self.body_quat_w_all[:, self.body_indexes]             # (N, B, 4)
        q_anchor = self._anchor_quat.unsqueeze(1)                       # (N, 1, 4)

        q_anchor_inv = quat_conjugate(q_anchor)                         # IsaacLab: unit quats → inverse = conjugate
        q_rel = quat_apply(q_anchor_inv, q_body)                        # broadcast quaternion multiply

        return q_rel.reshape(self.total_dataset_size, -1)

    @property
    def body_lin_vel_b(self):
        """
        body linear velocities in anchor-local frame.
        v_rel_local = R(q_anchor)^T (v_body - v_anchor)
        """
        v_body = self.body_lin_vel_w_all[:, self.body_indexes]          # (N, B, 3)
        v_anchor = self.anchor_lin_vel_w.unsqueeze(1)                   # (N, 1, 3)

        rel = v_body - v_anchor                                         # world frame
        rel_local = quat_apply_inverse(self._anchor_quat.unsqueeze(1), rel)

        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def body_ang_vel_b(self):
        """
        body angular velocities in anchor-local frame.
        ω_rel_local = R(q_anchor)^T (ω_body - ω_anchor)
        """
        w_body = self.body_ang_vel_w_all[:, self.body_indexes]          # (N, B, 3)
        w_anchor = self.anchor_ang_vel_w.unsqueeze(1)                   # (N, 1, 3)

        rel = w_body - w_anchor
        rel_local = quat_apply_inverse(self._anchor_quat.unsqueeze(1), rel)

        return rel_local.reshape(self.total_dataset_size, -1)

    
    @property
    def anchor_height(self):
        return self.anchor_pos_w[:, -1]
    
    @property
    def anchor_pos_w(self):
        return self.body_pos_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_quat_w(self):
        return self.body_quat_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
        
    @property
    def base_lin_vel(self):
        """
        Base (anchor) linear velocity expressed in base frame.
        Shape: (N, 3)
        """
        v_w = self.anchor_lin_vel_w                       # (N, 3)
        q_w = self.anchor_quat_w                          # (N, 4)

        v_b = quat_apply_inverse(q_w, v_w)                # world → base
        return v_b

    @property
    def base_ang_vel(self):
        """
        Base (anchor) angular velocity expressed in base frame.
        Shape: (N, 3)
        """
        w_w = self.anchor_ang_vel_w                       # (N, 3)
        q_w = self.anchor_quat_w                          # (N, 4)

        w_b = quat_apply_inverse(q_w, w_w)                # world → base
        return w_b


    # ----------------------- Transition index builder -----------------------

    def observation_dim_cast(self, name)->int:
        # shape_cast_table = {
        #     "displacement": self.body_indexes.shape[-1]
        # }
        if hasattr(self, name):
            obs_term: torch.Tensor = getattr(self, name)
            assert isinstance(obs_term, torch.Tensor), f"invalid observation name: {name} for get dim"
            return obs_term.shape[-1]
        else:
            raise NotImplementedError(f"Failed for term: {name}")

    def init_observation_dims(self):
        observation_dims = []
        for obs_term in self.observation_terms:
            # observation_terms.append(obs_term)
            observation_dims.append(self.observation_dim_cast(obs_term))
        self.observation_dim = sum(observation_dims)
        self.observation_dims = observation_dims

    # ----------------------- Transition index builder -----------------------

    def _build_transition_indices(self, traj_lengths: List[int], device: str):
        """
        Build valid (t, t+1) pairs without crossing trajectory boundaries.
        """
        idx_t = []
        idx_tp1 = []

        offset = 0
        for L in traj_lengths:
            if L < 2:
                offset += L
                continue
            t = torch.arange(offset, offset + L - 1)
            idx_t.append(t)
            idx_tp1.append(t + 1)
            offset += L

        idx_t = torch.cat(idx_t).to(device)
        idx_tp1 = torch.cat(idx_tp1).to(device)
        return idx_t, idx_tp1

    def _build_history_end_indices(self, traj_lengths: List[int], history_steps: int, device: str):
        """Build valid history-window end indices.

        对于每条轨迹，若长度为 L，窗口长度为 H，则可采样终点为 [H-1, ..., L-1]。
        """
        idx_end = []
        offset = 0
        for L in traj_lengths:
            if L >= history_steps:
                idx_end.append(torch.arange(offset + history_steps - 1, offset + L))
            offset += L
        if len(idx_end) == 0:
            raise ValueError(
                f"No valid discriminator history windows found. "
                f"Check motion length and history_steps={history_steps}."
            )
        return torch.cat(idx_end).to(device)

    def _build_history_indices_from_end(self, idx_end: torch.Tensor):
        """Convert end indices [B] to frame indices [B, H] (time order: old -> new)."""
        offsets = torch.arange(self.history_steps - 1, -1, -1, device=self.device, dtype=torch.long)
        return idx_end.unsqueeze(-1) - offsets.unsqueeze(0)

    def build_history_indices_from_end(self, idx_end: torch.Tensor):
        """Public wrapper for building history indices."""
        return self._build_history_indices_from_end(idx_end)

    # ----------------------- Batch Sampling API -----------------------

    def sample_batch(self, batch_size: int):
        """
        Sample a batch of transitions:
            s_t → s_{t+1}

        Returns dict:
            {
                "joint_pos_t": ...,
                "joint_pos_tp1": ...,
                ...
            }
        """
        idx = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
        t = self.index_t[idx]
        tp1 = self.index_tp1[idx]
        return t, tp1

    def sample_history_batch(self, batch_size: int):
        """Sample discriminator history windows.

        Returns:
            hist_idx: [B, H] global frame indices.
        """
        idx = torch.randint(0, len(self.index_hist_end), (batch_size,), device=self.device)
        idx_end = self.index_hist_end[idx]
        return self._build_history_indices_from_end(idx_end)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(0, num_mini_batch):
            hist_idx = self.sample_history_batch(mini_batch_size)
            yield self.build_history(hist_idx)
            
    def build_transition(self, t, tp1):
        res_t, res_tp1 = [], []
        for term in self.observation_terms:
            _t, _tp1 = getattr(self, term)[t], getattr(self, term)[tp1]
            res_t.append(_t); res_tp1.append(_tp1)
        res_t, res_tp1 = torch.cat(res_t, dim=-1), torch.cat(res_tp1, dim=-1)
        return res_t, res_tp1

    def build_history(self, hist_idx: torch.Tensor):
        """Build discriminator input from history indices.

        Args:
            hist_idx: [B, H] global frame indices.

        Returns:
            disc_obs: [B, H * D] where D=sum(term_dims)
        """
        res_hist = []
        for term in self.observation_terms:
            # term_data: [N, D_term] -> [B, H, D_term]
            term_data = getattr(self, term)[hist_idx]
            res_hist.append(term_data)
        # [B, H, D]
        hist = torch.cat(res_hist, dim=-1)
        return hist.reshape(hist.shape[0], -1)
        

@configclass
class MotionDatasetCfg:
    class_type          : type[MotionDataset] = MotionDataset
    asset_name          : str = "robot"
    motion_files        : List[str] = MISSING
    body_names          : List[str] = MISSING
    amp_obs_terms       : List[str] = MISSING
    anchor_name         : str = MISSING
    # 判别器历史帧窗口长度。2 等价旧版 (s_t, s_{t+1}) 拼接。
    history_steps       : int = 2
