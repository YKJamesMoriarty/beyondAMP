from amp_tasks.amp_task_demo_data_cfg import file_stance_idle
from robotlib.robot_keys.g1_29d import g1_anchor_name

experiment_name = "g1_stance_amp"

# 统一管理数据入口，方便后续替换多段 stance 动作文件。
amp_data_files = [file_stance_idle]

# 对齐 MimicKit G1 stance AMP 的关键 body 列表。
key_body_names = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    # 当前 IsaacLab G1 资产没有 head_link，使用 torso_link 作为上身关键点代理。
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
anchor_name = g1_anchor_name
