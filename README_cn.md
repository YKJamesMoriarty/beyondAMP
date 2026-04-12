# beyondAMP: One-Step Integration of AMP into IsaacLab

## 概览 (Overview) 🌍

**beyondAMP** 提供了一个统一的管道，将**对抗性运动先验 (Adversarial Motion Priors, AMP)** 集成到任何 IsaacLab 机器人设置中，只需进行最少的修改，并能完全兼容自定义的机器人设计。

-----

## 🚀 快速设置 (Fast Setup)

```bash
cd beyondAMP
bash scripts/setup_ext.sh
# Downloads assets, robot configs, and installs dependencies
```

可选的 VSCode 工作区设置：

```bash
python scripts/setup_vscode.py
```

-----

## 📌 如何使用 (How to Use)

### 快速开始 (Quick Start)

  * **基础环境 (Basic environment):** `source/amp_tasks/amp_tasks/amp`
  * **PPO 配置 (PPO config for G1 robot):** `source/amp_tasks/amp_tasks/amp/robots/g1/rsl_rl_ppo_cfg.py`

训练可以通过以下命令启动：

```bash
scripts/factoryIsaac/train.py --task AMPG1_Basic --headless
# scripts/factoryIsaac/train.py --task AMPG1_Soft --headless
# scripts/factoryIsaac/train.py --task AMPG1_Hard --headless
```

如果要训练 **stance 专属的纯 AMP 任务**（使用 `data/box/stance/trim_stance_orthodox_idle_normal_2_150.npz`）：

```bash
python scripts/factoryIsaac/train.py --task beyondAMP-StanceTask-G1-AMPBasic --headless --num_envs 128 --device cuda:0 --rldevice cuda:0
```

如果要记录到 Weights & Biases：

```bash
python scripts/factoryIsaac/train.py --task beyondAMP-StanceTask-G1-AMPBasic --headless --num_envs 128 --device cuda:0 --rldevice cuda:0 --logger wandb --log_project_name beyondamp
```

也可以在命令行直接覆盖训练轮数和保存间隔，例如：

```bash
python scripts/factoryIsaac/train.py --task beyondAMP-StanceTask-G1-AMPBasic --headless --num_envs 128 --device cuda:0 --rldevice cuda:0 --logger wandb --log_project_name beyondamp --max_iterations 100000 --save_interval 500
```

要评估或可视化一个已训练的检查点 (checkpoint)：

```bash
scripts/factoryIsaac/play.py --headless --target <path to your ckpt.pt> --video --num_envs 32
```

-----

### 数据集准备 (Dataset Preparation) 💾

The dataset follows the same structure and conventions used in BeyondMimic(whole\_body\_tracking). All motion sequences should be stored as **\*.npz** files and placed under `data/datasets/`, maintaining a consistent directory layout with the reference pipeline.

For motion retargeting and preprocessing, **GMR** is recommended for generating high-quality retargeted mocap data. **TrackerLab** may be used to perform forward kinematics checks and robot-specific adjustments, ensuring the motions remain physically plausible for your robot model.

With these tools, the dataset organization naturally aligns with the conventions established in BeyondMimic(whole\_body\_tracking), enabling seamless integration with the AMP training pipeline.

> **遵循 BeyondMimic 的数据集管道 (Following the dataset pipeline of BeyondMimic):**
>
>   * 动作文件 (Motion files): 将 `*.npz` 放入 `data/datasets/`
>   * **推荐工具 (Recommended tools):**
>       * **GMR** 用于重定向动作 (retargeted motion)
>       * **TrackerLab** 用于正向运动学 (FK) 验证和机器人特定的预处理

-----

### AMP 集成细节 (AMP Integration Details)

  * AMP observation group added via a new `amp` observation config
  * RSL-RL integration: `source/rsl_rl/rsl_rl/env/isaaclab/amp_wrapper.py`
  * Default transition builder: `source/beyondAMP/beyondAMP/amp_obs.py`

> 有关完整的教程和自定义，请参阅 (For full tutorial and customization, see) `docs/tutorial.md`。

\<details\>
\<summary\>\<strong\>附加说明 (Additional Notes)\</strong\>\</summary\>

  * 完全模块化的 AMP 观察构建器 (Fully modular AMP observation builder)
  * 兼容 IsaacLab 4.5+ (Compatible with IsaacLab 4.5+)
  * 专为跨机器人形态的快速实验而设计 (Designed for rapid experimentation across robot morphologies)

\</details\>

-----

## 🙏 致谢 (Acknowledgement)

### 参考仓库 (Referenced Repositories)

| 仓库 (Repository)                                                           | 用途 (Purpose)                               |
| ----------------------------------------------------------------------------- | --------------------------------------------- |
| [robotlib](https://github.com/Renforce-Dynamics/robotlib)                     | 机器人配置 (Robot configurations)                     |
| [assetslib](https://github.com/Renforce-Dynamics/assetslib)                   | 资产存储 (Asset storage)                         |
| [TrackerLab](https://github.com/Renforce-Dynamics/trackerLab)                 | 数据组织与重定向工具 (Data organization & retargeting tools) |
| [AMP\_for\_hardware](https://github.com/escontra/AMP_for_hardware)          | AMP 实现参考 (AMP implementation reference)          |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)       | 数据集格式与跟踪比较 (Dataset format & tracking comparison)  |

-----

## 📘 引用 (Citation)

```bibtex
@software{zheng2025@beyondAMP,
  author = {Ziang Zheng},
  title = {beyondAMP: One step unify IsaacLab with AMP.},
  url = {https://github.com/Renforce-Dynamics/beyondAMP},
  year = {2025}
}
```
