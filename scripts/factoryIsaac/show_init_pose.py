"""可视化指定任务的“默认初始姿态”（不依赖训练模型）。"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Show task initial pose (debug utility).")
parser.add_argument("--task", type=str, default="beyondAMP-StanceTask-G1-AMPBasic", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--steps", type=int, default=600, help="Simulation steps to run.")
parser.add_argument("--video", action="store_true", default=False, help="Record a short video.")
parser.add_argument("--video_length", type=int, default=300, help="Recorded video length.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import amp_tasks  # noqa: F401  # 注册任务


def _freeze_to_default_init_pose(env_cfg):
    """将 reset 行为改为“严格默认姿态”，仅用于可视化调试。"""
    # 不从参考动作库随机帧重置
    if hasattr(env_cfg.events, "reset_to_ref_motion_dataset"):
        env_cfg.events.reset_to_ref_motion_dataset = None

    # 关闭 interval 外力推搡，避免调试画面被扰动
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # root reset: 默认 root 状态 + 0 扰动
    if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    # joints reset: 默认关节状态 * 1.0（不随机缩放）
    if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (1.0, 1.0)

    return env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg = _freeze_to_default_init_pose(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(".", "videos", "show_init_pose"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    _ = obs

    action_shape = env.action_space.sample().shape
    zero_actions = torch.zeros(action_shape, device=env.unwrapped.device)

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        with torch.inference_mode():
            env.step(zero_actions)
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
