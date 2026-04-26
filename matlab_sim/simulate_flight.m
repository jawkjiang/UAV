function [t, state9d] = simulate_flight(mission, rng_state)
%SIMULATE_FLIGHT  Simulate UAV sensor readings for one waypoint mission.
%
%   [t, state9d] = simulate_flight(mission, rng_state)
%
%   Inputs:
%     mission  : struct from generate_mission()
%     rng_state: integer seed
%
%   Outputs:
%     t        : [M×1] time vector [s] at output sample rate FS
%     state9d  : [M×9] sensor state matrix
%                columns: [x, y, z,   vx, vy, vz,   ax, ay, az]
%                units:    m, m, m,   m/s, m/s, m/s, m/s², m/s², m/s²
%                frame: ENU

rng(rng_state + 1000, 'twister');   % offset to differ from mission seed
config;

waypoints = mission.waypoints;
dur       = mission.duration;
n_wp      = size(waypoints, 1);

%% Build fine-grained true trajectory (position/velocity at SIM_FS)
dt_sim = 1 / SIM_FS;
t_sim  = (0 : dt_sim : dur)';
N_sim  = length(t_sim);

% Distribute time proportionally to cumulative segment length
seg_len    = zeros(n_wp, 1);
seg_len(1) = 0;
for k = 2:n_wp
    seg_len(k) = seg_len(k-1) + norm(waypoints(k,:) - waypoints(k-1,:));
end
total_len = max(seg_len(end), 1e-6);   % avoid div-by-zero for trivial missions
t_wp = seg_len / total_len * dur;      % [n_wp × 1], starts at 0, ends at dur

% Interpolate true position along waypoint path
pos_true = zeros(N_sim, 3);
for dim = 1:3
    pos_true(:, dim) = interp1(t_wp, waypoints(:, dim), t_sim, 'pchip', 'extrap');
end

% Velocity via central differences (clamp endpoints)
vel_true = gradient_nd(pos_true, dt_sim);

% Acceleration via central differences of velocity
acc_true = gradient_nd(vel_true, dt_sim);

%% Add sensor noise

% GPS position noise (at SIM_FS, will decimate later)
gps_pos = pos_true + GPS_HPOS_SIGMA * randn(N_sim, 2, 'double');
gps_pos = [gps_pos, pos_true(:,3) + GPS_VPOS_SIGMA * randn(N_sim, 1)];

% GPS velocity noise
gps_vel = vel_true + GPS_VEL_SIGMA * randn(N_sim, 3);

% IMU acceleration (body-frame noise; assume level flight so body ≈ ENU)
imu_acc = acc_true + IMU_ACCEL_SIGMA * randn(N_sim, 3);

% --- Combine into 9-d state vector at SIM_FS ---
state_sim = [gps_pos, gps_vel, imu_acc];

%% Decimate to FS output rate
decim = round(SIM_FS / FS);
idx   = 1 : decim : N_sim;
t       = t_sim(idx);
state9d = state_sim(idx, :);
end

%% Helper ---------------------------------------------------------------
function out = gradient_nd(X, dt)
% Central differences for each column of X
out = zeros(size(X));
out(2:end-1, :) = (X(3:end,:) - X(1:end-2,:)) / (2*dt);
out(1,       :) = (X(2,:)     - X(1,:))        / dt;
out(end,     :) = (X(end,:)   - X(end-1,:))    / dt;
end
