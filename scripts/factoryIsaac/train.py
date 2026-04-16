import argparse
import os
import re
from isaaclab import __version__ as omni_isaac_lab_version
assert omni_isaac_lab_version > "0.21.0"
from isaaclab.app import AppLauncher
# local imports
import argtool as rsl_arg_cli  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--alg", type=str, default="PPO", help="Name of the algorithm.")
parser.add_argument("--cfg", type=str, default=None, help="Directly using the target cfg object.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--replicate", type=str, default=None, help="Replicate old experiment with same configuration.")

parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--local", action="store_true", default=False, help="Using asset in local buffer")
parser.add_argument(
    "--wandb_path",
    type=str,
    default=None,
    help="Wandb run path or checkpoint path, e.g. entity/project/run_id or entity/project/run_id/model_50000.pt",
)

# append RSL-RL cli arguments
rsl_arg_cli.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import amp_tasks

from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner


def _resolve_wandb_checkpoint(wandb_path: str) -> tuple[str, str, str]:
    """从 wandb 解析并下载 checkpoint。

    Returns:
        (run_path, model_file_name, local_checkpoint_path)
    """
    import wandb

    query_path = wandb_path.strip().rstrip("/")
    target_file = None

    # 支持直接传 entity/project/run_id/model_xxx.pt
    if query_path.endswith(".pt"):
        target_file = os.path.basename(query_path)
        query_path = os.path.dirname(query_path)

    api = wandb.Api()
    run = api.run(query_path)
    run_files = list(run.files())

    chosen_file = None
    if target_file is not None:
        for f in run_files:
            if os.path.basename(f.name) == target_file:
                chosen_file = f
                break
        if chosen_file is None:
            raise FileNotFoundError(f"Cannot find checkpoint '{target_file}' in wandb run: {query_path}")
    else:
        # 自动选最大的 model_{iter}.pt
        model_candidates = []
        for f in run_files:
            matched = re.search(r"model_(\d+)\.pt$", os.path.basename(f.name))
            if matched is not None:
                model_candidates.append((int(matched.group(1)), f))
        if len(model_candidates) == 0:
            raise FileNotFoundError(f"No checkpoint like model_*.pt found in wandb run: {query_path}")
        model_candidates.sort(key=lambda x: x[0])
        chosen_file = model_candidates[-1][1]

    run_id = query_path.split("/")[-1]
    download_root = os.path.abspath(os.path.join("logs", "rsl_rl", "temp", run_id))
    os.makedirs(download_root, exist_ok=True)

    downloaded_fp = chosen_file.download(root=download_root, replace=True)
    local_checkpoint = os.path.abspath(downloaded_fp.name)
    downloaded_fp.close()

    return query_path, chosen_file.name, local_checkpoint


def main():
    task_name, env_cfg, agent_cfg, log_dir = rsl_arg_cli.make_cfgs(args_cli, parse_env_cfg, None)
        
    # env_cfg.scene.terrain.terrain_generator.num_cols = 2
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # create isaac environment
    env = gym.make(task_name, cfg=env_cfg, 
                   render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.rldevice
    
    env, func_runner, learn_cfg = rsl_arg_cli.prepare_wrapper(env, args_cli, agent_cfg)
    runner: AMPOnPolicyRunner = func_runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # save resume path before creating a new log_dir
    # 兼容两种续训来源：
    # 1) 本地 --checkpoint
    # 2) 远端 --wandb_path（自动下载后加载）
    if args_cli.resume or (args_cli.wandb_path is not None):
        if args_cli.wandb_path is not None:
            run_path_wandb, model_file_name, resume_path = _resolve_wandb_checkpoint(args_cli.wandb_path)
            print(f"[INFO]: Downloaded wandb checkpoint: {run_path_wandb}/{model_file_name}")
        else:
            resume_path = args_cli.checkpoint

        if resume_path is None:
            raise ValueError("Resume requires --checkpoint or --wandb_path.")

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "args.yaml"), vars(args_cli))
    rsl_arg_cli.dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    rsl_arg_cli.dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    rsl_arg_cli.dump_pickle(os.path.join(log_dir, "params", "args.pkl"), args_cli)

    # run training
    try:
        runner.learn(**learn_cfg)
    finally:
        # 统一释放日志后端资源（如 wandb run / tensorboard writer）。
        if hasattr(runner, "close"):
            runner.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
