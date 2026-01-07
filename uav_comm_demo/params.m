% params.m - 参数配置文件

% 节点位置（单位：米）
user_pos = [0, 0];           % 地面用户
jammer_pos = [100, 0];       % 干扰源
bs_pos = [300, 0];           % 基站
uav_init_pos = [150, 100];   % UAV 初始位置（空中）

% 模拟参数
Pt = 1;                      % 发射功率（单位：W）
Pj = 1;                      % 干扰功率（单位：W）
N0 = 1e-9;                   % 噪声功率
sim_time = 100;              % 仿真时间步数
v_uav = 1;                   % UAV 飞行速度（米/步）
sinr_threshold = 5;          % SINR 阈值（dB）
