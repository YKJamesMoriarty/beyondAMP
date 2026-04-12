from amp_tasks.amp_task_demo_data_cfg import file_stance_idle
from robotlib.robot_keys.g1_29d import g1_anchor_name, g1_key_body_names

experiment_name = "g1_stance_amp"

# 统一管理数据入口，方便后续替换多段 stance 动作文件。
amp_data_files = [file_stance_idle]

key_body_names = g1_key_body_names
anchor_name = g1_anchor_name
