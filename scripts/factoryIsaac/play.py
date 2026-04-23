"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import tqdm
import re
import sys
import numpy as np
from isaaclab import __version__ as omni_isaac_lab_version
from isaaclab.app import AppLauncher

# local imports
import argtool as rsl_arg_cli  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
if omni_isaac_lab_version < "0.21.0":
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--target", type=str, default=None, help="If use, direct point to the target ckpt")
parser.add_argument(
    "--wandb_path",
    type=str,
    default=None,
    help="Wandb run path or checkpoint path, e.g. entity/project/run_id or entity/project/run_id/model_50000.pt",
)
parser.add_argument(
    "--onnx_path",
    type=str,
    default=None,
    help="Path to an exported ONNX policy, e.g. logs/.../exported/policy.onnx",
)

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--run", type=str, default=".*", help="Name of the run.")
# parser.add_argument("--ckpt", type=str, default=".*", help="Name of the ckpt.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")

parser.add_argument("--collect", action="store_true", default=False, help="Record data during playing.")
parser.add_argument("--web", action="store_true", default=False, help="Web videos during playing.")
parser.add_argument("--local", action="store_true", default=False, help="Using asset in local buffer")

parser.add_argument("--determine",action="store_true", default=False, help="Clear reset terms" )

# append RSL-RL cli arguments
rsl_arg_cli.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video or args_cli.web:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import os
import torch

import amp_tasks
# import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.dict import print_dict

# Import extensions to set up environment tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from rsl_rl_amp.runners import OnPolicyRunner
from beyondAMP.isaaclab.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx


def _make_onnx_policy(onnx_path: str):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "ONNX playback requires onnxruntime. Install it in the same Python environment that launches Isaac Sim. "
            f"Current python: {sys.executable}"
        ) from exc

    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = [item.name for item in session.get_inputs()]
    if len(input_names) != 1:
        raise RuntimeError(
            f"Current play path only supports single-input ONNX policies, but got inputs={input_names}."
        )
    output_names = [item.name for item in session.get_outputs()]
    return session, input_names[0], output_names[0]


def _onnx_policy_act(session, input_name: str, obs: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    obs_np = obs.detach().to("cpu", copy=True).numpy().astype(np.float32, copy=False)
    if obs_np.ndim == 1:
        obs_np = obs_np[None, :]

    input_shape = session.get_inputs()[0].shape
    fixed_batch = len(input_shape) > 0 and isinstance(input_shape[0], int) and input_shape[0] == 1

    if obs_np.shape[0] == 1 or not fixed_batch:
        actions_np = session.run(None, {input_name: obs_np})[0]
    else:
        # The exported policy in this project is typically traced with batch=1.
        # To keep vectorized environments working, run the ONNX policy env-by-env.
        actions_list = []
        for i in range(obs_np.shape[0]):
            action_i = session.run(None, {input_name: obs_np[i : i + 1]})[0]
            actions_list.append(action_i[0])
        actions_np = np.stack(actions_list, axis=0)

    return torch.from_numpy(actions_np).to(device=device)


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
    """Play with RSL-RL agent. base branch"""
    task_name = args_cli.task
    use_onnx = args_cli.onnx_path is not None
    if use_onnx:
        resume_path = os.path.abspath(args_cli.onnx_path)
        log_root_path = os.path.dirname(os.path.dirname(resume_path))
        print(f"[INFO]: Loading ONNX policy from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    elif args_cli.wandb_path is not None:
        run_path_wandb, model_file_name, resume_path = _resolve_wandb_checkpoint(args_cli.wandb_path)
        log_root_path = os.path.dirname(os.path.dirname(resume_path))
        print(f"[INFO]: Downloaded wandb checkpoint: {run_path_wandb}/{model_file_name}")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    elif args_cli.target is None:
        raise ValueError("Please provide either --target or --wandb_path.")
    else:
        resume_path = os.path.abspath(args_cli.target)
        log_root_path = os.path.dirname(os.path.dirname(resume_path))
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    
    if args_cli.collect:
        sample_dir = os.path.join(run_path, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        output_file = os.path.join(sample_dir, "total_data.pkl")
    
    if task_name is None:
        if args_cli.wandb_path is not None:
            raise ValueError("When using --wandb_path, please specify --task explicitly.")
        assert os.path.exists(os.path.join(run_path, "params", "args.pkl")), "No task specified."
        with open(os.path.join(run_path, "params", "args.pkl"), "rb") as f:
            args_old = pickle.load(f)
        task_name = args_old.task

        with open(os.path.join(run_path, "params", "env.pkl"), "rb") as f:
            env_cfg = pickle.load(f)
            
        with open(os.path.join(run_path, "params", "agent.pkl"), "rb") as f:
            agent_cfg = pickle.load(f)
    else:
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        agent_cfg = rsl_arg_cli.parse_rsl_rl_cfg(task_name, args_cli, None)

    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # env_cfg.commands.punch_command.debug_vis = True

    from isaaclab.envs.common import ViewerCfg
    
    
    env_cfg.viewer = ViewerCfg(
        eye = (4.0, 4.0, 4.0),
        # eye = (0.0, 0.0, 10.0),
        lookat = (0.0, 0.0, 0.0),
        env_index = 20,
        origin_type = "asset_root",
        # origin_type = "env",
        asset_name = "robot",
    )
    
    env_cfg.curriculum = None
    # if args_cli.determine:
    #     set_determine_reset(env_cfg)
    
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    
    # env_cfg.commands.base_velocity.debug_vis = False
    # env_cfg.scene.height_scanner.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    render_mode = "rgb_array" if args_cli.video or args_cli.web else None
    # create isaac environment
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.length if args_cli.length > 0 else 0,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        # print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.rldevice

    env, func_runner, learn_cfg = rsl_arg_cli.prepare_wrapper(env, args_cli, agent_cfg)

    onnx_session = None
    onnx_input_name = None
    policy = None

    if use_onnx:
        onnx_session, onnx_input_name, onnx_output_name = _make_onnx_policy(resume_path)
        print(f"[INFO]: Using ONNX output node: {onnx_output_name}")
    else:
        runner:OnPolicyRunner = func_runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

        runner.load(resume_path)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs = env.get_observations()

    if not use_onnx:
        # export policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        os.makedirs(export_model_dir, exist_ok=True)
        torch.save(runner.alg.actor_critic, os.path.join(export_model_dir, "policy.pth"))
        export_policy_as_onnx(runner.alg.actor_critic, export_model_dir, filename="policy.onnx")
        export_policy_as_jit(runner.alg.actor_critic, None, export_model_dir, filename="policy.pt")
        # export_policy_as_jit(runner.alg.policy, runner.alg.policy.actor_obs_normalizer, export_model_dir, filename="policy.pt")

        print(f"[INFO]: Saving policy to: {export_model_dir}")

    pbar = tqdm.tqdm(range(args_cli.length)) if args_cli.length>0  else tqdm.tqdm()
    
    step = 0
    try:
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                if use_onnx:
                    actions = _onnx_policy_act(onnx_session, onnx_input_name, obs, env.unwrapped.device)
                else:
                    actions = policy(obs)
                # env stepping
                obs, rewards, dones, infos = env.step(actions, not_amp=True)

            step += 1
            pbar.update() 
            if args_cli.collect:
                # trans = [obs.data, actions.data, rewards.data, dones.data, ppo_runner.alg.actor_critic.laction.data]
                trans = [obs.data, actions.data, rewards.data, dones.data]
                with open(output_file, 'ab') as f:
                    pickle.dump(trans, f)
            if args_cli.length > 0 and args_cli.length < step:
                break

    except KeyboardInterrupt:
        pass

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
