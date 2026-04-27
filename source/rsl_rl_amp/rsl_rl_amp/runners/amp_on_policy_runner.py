import time
import os
from collections import deque
import statistics

import numpy as np
import torch

from rsl_rl_amp.algorithms import AMPPPO, PPO, AMPPPOWeighted
from rsl_rl_amp.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl_amp.env import VecEnv
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer
from beyondAMP.isaaclab.rsl_rl.amp_wrapper import AMPEnvWrapper
from .logger_backend import ScalarLogger

from beyondAMP.motion.motion_dataset import MotionDataset

class AMPOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.amp_data_cfg = train_cfg["amp_data"]
        self.device = device
        self.env:AMPEnvWrapper = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.policy_cfg["class_name"]) # ActorCritic
        num_actor_obs = self.env.num_obs
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)

        amp_data = env.motion_dataset
        amp_obs_dim = env.get_amp_observations().shape[-1] # amp_data.observation_dim
        self.amp_obs_dim = amp_obs_dim
        self.amp_obs_terms = list(amp_data.observation_terms)
        self.amp_obs_term_slices = self._build_obs_term_slices(self.amp_obs_terms, list(amp_data.observation_dims))
        self.amp_obs_history_steps = max(int(self.amp_data_cfg.get("history_steps", 2)), 2)
        self.root_xy_pos_rel_to_latest = bool(self.amp_data_cfg.get("root_xy_pos_rel_to_latest", False))
        self.key_body_pos_rel_to_root = bool(self.amp_data_cfg.get("key_body_pos_rel_to_root", False))
        self.amp_disc_obs_dim = self.amp_obs_dim * self.amp_obs_history_steps
        amp_normalizer = Normalizer(self.amp_disc_obs_dim)
        discriminator = AMPDiscriminator(
            self.amp_disc_obs_dim,
            train_cfg['amp_reward_coef'],
            train_cfg['amp_discr_hidden_dims'], device,
            train_cfg['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.alg_cfg["class_name"]) # PPO
        min_std = (
            torch.tensor(self.cfg["amp_min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[0, :, 1] - self.env.dof_pos_limits[0, :, 0])))
        self.alg: AMPPPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = str(self.cfg.get("logger", "tensorboard"))
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
        self._amp_obs_history = None

    @staticmethod
    def _build_obs_term_slices(term_names, term_dims):
        term_slices = {}
        start = 0
        for name, dim in zip(term_names, term_dims):
            term_slices[name] = slice(start, start + dim)
            start += dim
        return term_slices

    def _init_amp_history(self, amp_obs: torch.Tensor):
        """使用当前观测初始化历史窗口 [N, H, D]。"""
        self._amp_obs_history = amp_obs.unsqueeze(1).repeat(1, self.amp_obs_history_steps, 1)

    def _push_amp_history(self, next_amp_obs: torch.Tensor):
        """历史窗口左移一帧，并写入最新 amp 观测。"""
        self._amp_obs_history = torch.roll(self._amp_obs_history, shifts=-1, dims=1)
        self._amp_obs_history[:, -1, :] = next_amp_obs

    def _build_disc_obs_from_history(self, amp_obs_hist: torch.Tensor):
        """将历史窗口观测转换为判别器输入（贴近 MimicKit 语义）。

        输入:
            amp_obs_hist: [N, H, D_raw]
        输出:
            disc_obs: [N, H * D_raw]
        """
        hist = amp_obs_hist.clone()
        root_pos_raw = None

        # 读取 root（anchor）绝对位置
        if "root_pos_w" in self.amp_obs_term_slices:
            root_slice = self.amp_obs_term_slices["root_pos_w"]
            root_pos_raw = hist[:, :, root_slice]  # [N,H,3]

        # root position: x/y 相对历史窗口最新帧，z 保持绝对值。
        if self.root_xy_pos_rel_to_latest:
            if root_pos_raw is None:
                raise ValueError("`root_xy_pos_rel_to_latest=True` requires `root_pos_w` in amp_obs_terms.")
            if root_pos_raw.shape[-1] != 3:
                raise ValueError(f"`root_pos_w` dim must be 3, got {root_pos_raw.shape[-1]}.")
            ref_xy = root_pos_raw[:, -1:, 0:2]
            root_obs = root_pos_raw.clone()
            root_obs[..., 0:2] = root_obs[..., 0:2] - ref_xy
            hist[:, :, root_slice] = root_obs

        # key body position relative root: key_pos_w - root_pos_w（逐帧）。
        if self.key_body_pos_rel_to_root:
            if "body_pos_w" not in self.amp_obs_term_slices:
                raise ValueError("`key_body_pos_rel_to_root=True` requires `body_pos_w` in amp_obs_terms.")
            if root_pos_raw is None:
                raise ValueError("`key_body_pos_rel_to_root=True` requires `root_pos_w` in amp_obs_terms.")
            key_slice = self.amp_obs_term_slices["body_pos_w"]
            key_pos_w = hist[:, :, key_slice]
            if key_pos_w.shape[-1] % 3 != 0:
                raise ValueError(f"`body_pos_w` dim must be multiple of 3, got {key_pos_w.shape[-1]}.")
            num_key = key_pos_w.shape[-1] // 3
            key_pos_w = key_pos_w.view(key_pos_w.shape[0], key_pos_w.shape[1], num_key, 3)
            key_pos_rel = key_pos_w - root_pos_raw.unsqueeze(-2)
            hist[:, :, key_slice] = key_pos_rel.reshape(key_pos_rel.shape[0], key_pos_rel.shape[1], -1)

        return hist.reshape(hist.shape[0], -1)

    def _reset_amp_history_for_envs(self, env_ids: torch.Tensor, reset_obs: torch.Tensor):
        """done 环境进入新 episode 时，将历史窗口重置为当前 reset 观测。"""
        if env_ids.numel() == 0:
            return
        fill_obs = reset_obs[env_ids].unsqueeze(1).repeat(1, self.amp_obs_history_steps, 1)
        self._amp_obs_history[env_ids] = fill_obs

    def _inject_terminal_amp_obs(self, hist_obs: torch.Tensor, reset_env_ids: torch.Tensor, terminal_amp_states: torch.Tensor):
        """将 done 环境最后一帧（最新帧）替换为终止观测。"""
        if reset_env_ids.numel() == 0:
            return hist_obs
        term_hist = hist_obs.clone()
        terminal_amp_states = terminal_amp_states.to(hist_obs.device)
        term_hist[reset_env_ids, -1, :] = terminal_amp_states
        return term_hist
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # 统一通过日志后端封装，支持 tensorboard / wandb。
            self.writer = ScalarLogger(self.logger_type, self.log_dir, self.cfg)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        self._init_amp_history(amp_obs.to(self.device))
        amp_disc_obs = self._build_disc_obs_from_history(self._amp_obs_history)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_disc_obs = obs.to(self.device), critic_obs.to(self.device), amp_disc_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        ampbuffer = deque(maxlen=100)
        discribuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_discri_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) 
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_disc_obs)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions, not_amp=False)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # 更新历史窗口，并构造判别器输入。
                    self._push_amp_history(next_amp_obs)
                    next_hist = self._amp_obs_history
                    next_hist_with_term = self._inject_terminal_amp_obs(
                        next_hist, reset_env_ids, terminal_amp_states
                    )
                    next_amp_disc_obs_with_term = self._build_disc_obs_from_history(next_hist_with_term)

                    lerp_rewards, d_logits, amp_rewards = self.alg.discriminator.predict_amp_reward(
                        state=next_amp_disc_obs_with_term,
                        task_reward=rewards,
                        normalizer=self.alg.amp_normalizer,
                    )
                    d_logits = d_logits.squeeze(-1)
                    amp_rewards = amp_rewards.squeeze(-1)
                    self.alg.process_env_step(lerp_rewards, dones, infos, next_amp_disc_obs_with_term)
                    # 进入新 episode 的环境，历史窗口应从 reset 状态重新开始。
                    self._reset_amp_history_for_envs(reset_env_ids, next_amp_obs)
                    amp_disc_obs = torch.clone(self._build_disc_obs_from_history(self._amp_obs_history))
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_amp_sum += amp_rewards
                        cur_discri_sum += d_logits
                        cur_episode_length += 1
                        self.log_loc(cur_reward_sum, dones, rewbuffer)
                        self.log_loc(cur_amp_sum, dones, ampbuffer)
                        self.log_loc(cur_discri_sum, dones, discribuffer)
                        self.log_loc(cur_episode_length, dones, lenbuffer)

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, \
            mean_amp_loss, mean_grad_pen_loss, \
            mean_policy_pred, mean_expert_pred, \
            mean_disc_agent_acc, mean_disc_demo_acc, \
            mean_disc_margin = \
                self.alg.update()
                
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        self.close()

    def log_loc(self, cur_sum, dones, buffer):
        new_ids = (dones > 0).nonzero(as_tuple=False)
        buffer.extend(cur_sum[new_ids][:, 0].cpu().numpy().tolist())
        cur_sum[new_ids] = 0

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        # 判别器二分类准确率：agent 判负类、demo 判正类。
        self.writer.add_scalar('Disc_Agent_Acc', locs['mean_disc_agent_acc'], locs['it'])
        self.writer.add_scalar('Disc_Demo_Acc', locs['mean_disc_demo_acc'], locs['it'])
        self.writer.add_scalar('Disc_Margin', locs['mean_disc_margin'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/disc_learning_rate', getattr(self.alg, "disc_learning_rate", self.alg.learning_rate), locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            # 注意：wandb 需要 step 全局单调递增。这里统一用 iteration 作为 step，
            # 避免和 time 轴混用导致 "step less than current step" 告警与数据被丢弃。
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_amp_reward', statistics.mean(locs['ampbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_discri_logits', statistics.mean(locs['discribuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_amp_reward/time', statistics.mean(locs['ampbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_discri_logits/time', statistics.mean(locs['discribuffer']), locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Disc Agent Acc:':>{pad}} {locs['mean_disc_agent_acc']:.4f}\n"""
                          f"""{'Disc Demo Acc:':>{pad}} {locs['mean_disc_demo_acc']:.4f}\n"""
                          f"""{'Disc Margin:':>{pad}} {locs['mean_disc_margin']:.4f}\n"""
                          f"""{'Policy learning rate:':>{pad}} {self.alg.learning_rate:.2e}\n"""
                          f"""{'Disc learning rate:':>{pad}} {getattr(self.alg, 'disc_learning_rate', self.alg.learning_rate):.2e}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        payload = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        # 兼容新增的判别器独立优化器。
        if hasattr(self.alg, "disc_optimizer") and self.alg.disc_optimizer is not None:
            payload['disc_optimizer_state_dict'] = self.alg.disc_optimizer.state_dict()
        torch.save(payload, path)
        # 可选：上传 checkpoint 到 wandb（tensorboard 模式下为 no-op）。
        if self.writer is not None:
            self.writer.save_file(path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            # 兼容旧 checkpoint（仅有 optimizer_state_dict）和新 checkpoint（含 disc optimizer）。
            if 'optimizer_state_dict' in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if hasattr(self.alg, "disc_optimizer") and self.alg.disc_optimizer is not None:
                if 'disc_optimizer_state_dict' in loaded_dict:
                    self.alg.disc_optimizer.load_state_dict(loaded_dict['disc_optimizer_state_dict'])
        if 'iter' in loaded_dict:
            self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def close(self):
        """关闭日志资源（可重复调用）。"""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
