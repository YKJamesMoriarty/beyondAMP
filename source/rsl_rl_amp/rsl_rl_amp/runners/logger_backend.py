from __future__ import annotations

import os
from typing import Any

import torch


def _to_float(value: Any) -> float:
    """将标量值稳健转换为 float，便于统一写入日志后端。"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.float().mean().item())
    return float(value)


def _to_jsonable(value: Any) -> Any:
    """将配置对象递归转为 wandb 可接受的 json 结构。

    对无法序列化的对象（如 class/type/callable）统一转字符串，避免 wandb init 失败。
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


class ScalarLogger:
    """训练标量日志统一封装。

    支持:
    - tensorboard
    - wandb

    说明:
    - 本类只处理标量，避免把 runner 的训练逻辑与具体日志平台强耦合。
    - 若用户选择的日志后端不可用，会抛出明确报错，便于快速排查。
    """

    def __init__(self, logger_type: str, log_dir: str, train_cfg: dict):
        self.logger_type = (logger_type or "tensorboard").lower()
        self.log_dir = log_dir
        self._writer = None
        self._wandb = None
        self._wandb_run = None

        if self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
            return

        if self.logger_type == "wandb":
            try:
                import wandb
            except ImportError as err:
                raise ImportError(
                    "logger='wandb' 但当前环境未安装 wandb。请先执行: pip install wandb"
                ) from err

            self._wandb = wandb
            os.makedirs(log_dir, exist_ok=True)

            run_name = os.path.basename(log_dir.rstrip("/"))
            project_name = str(train_cfg.get("wandb_project", "isaaclab"))
            cfg_json = _to_jsonable(train_cfg)

            self._wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                dir=log_dir,
                config=cfg_json,
                reinit=False,
            )
            return

        if self.logger_type == "neptune":
            raise NotImplementedError(
                "当前 rsl_rl_amp runner 尚未实现 neptune 后端，请改用 --logger tensorboard 或 --logger wandb。"
            )

        raise ValueError(f"Unsupported logger type: {self.logger_type}")

    def add_scalar(self, tag: str, value: Any, step: Any):
        scalar_value = _to_float(value)
        if self.logger_type == "tensorboard":
            self._writer.add_scalar(tag, scalar_value, step)
            return

        # wandb 的 step 使用 int 更稳定；非整型步长统一四舍五入。
        step_int = int(round(float(step)))
        self._wandb.log({tag: scalar_value}, step=step_int)

    def close(self):
        if self.logger_type == "tensorboard" and self._writer is not None:
            self._writer.close()
            self._writer = None
            return
        if self.logger_type == "wandb" and self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def save_file(self, file_path: str):
        """将本地文件归档到日志后端。

        - tensorboard: 无操作（本身不负责文件托管）
        - wandb: 显式上传文件（如 checkpoint）
        """
        if self.logger_type != "wandb":
            return
        if self._wandb_run is None:
            return
        # 使用 base_path 保留相对目录结构，便于在 wandb 上定位文件。
        self._wandb.save(file_path, base_path=os.path.dirname(file_path), policy="live")
