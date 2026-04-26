function [state9d_attacked, attack_info] = inject_attack(t, state9d, attack_type, params, rng_state)
%INJECT_ATTACK  Apply a GPS spoofing attack to a flight's sensor readings.
%
%   [state9d_attacked, attack_info] = inject_attack(t, state9d, attack_type, params, rng_state)
%
%   Inputs:
%     t           : [M×1] time vector [s]
%     state9d     : [M×9] clean sensor state
%     attack_type : 'step' | 'drift' | 'delay' | 'takeover'
%     params      : struct with attack-specific fields (see below)
%     rng_state   : integer for reproducibility
%
%   Outputs:
%     state9d_attacked : [M×9] state after attack injection
%     attack_info      : struct with onset_time, attack_type, params
%
%  Shared params:
%    params.onset_frac  : fraction of total duration for attack onset [0,1]
%    params.direction   : 'random_xy'|'along_track_xy'|'cross_track_xy'|
%                          'fixed_east'|'fixed_north'
%
%  Type-specific params:
%    'step'    : params.M           offset magnitude [m]
%    'drift'   : params.M           final offset [m]
%                params.T_drift     drift duration [s]
%    'delay'   : params.delta       time lag [s]
%    'takeover': params.M           final offset [m]
%                params.alpha       tracking gain [0,1]

config;
rng(rng_state + 2000, 'twister');

M = length(t);
dur = t(end) - t(1);

% ---- compute attack onset ----
onset_frac = params.onset_frac;
t_s = t(1) + onset_frac * dur;
t_s = max(t(1) + 2.0, min(t(end) - 2.0, t_s));  % safety buffer
onset_idx = find(t >= t_s, 1, 'first');

% ---- direction vector ----
d_hat = compute_direction(params.direction, state9d, onset_idx);

% ---- extract position columns ----
pos_attacked = state9d(:, 1:3);
vel_attacked = state9d(:, 4:6);
acc_attacked = state9d(:, 7:9);

switch lower(attack_type)

    case 'step'
        % Instantaneous position offset
        M_off = params.M;
        offset_2d = M_off * d_hat;
        for k = onset_idx:M
            pos_attacked(k, 1:2) = pos_attacked(k, 1:2) + offset_2d';
        end
        % Transient velocity disturbance
        [vel_attacked, acc_attacked] = add_velocity_transient(...
            t, vel_attacked, acc_attacked, t_s, d_hat, 1.5, 1.0, 5.0);

    case 'drift'
        % Linearly growing offset over T_drift seconds
        M_off  = params.M;
        T_drift = params.T_drift;
        for k = onset_idx:M
            dt_k = t(k) - t_s;
            g    = min(1.0, dt_k / T_drift);
            pos_attacked(k, 1:2) = pos_attacked(k, 1:2) + (M_off * g * d_hat)';
        end
        [vel_attacked, acc_attacked] = add_velocity_transient(...
            t, vel_attacked, acc_attacked, t_s, d_hat, 1.5, 1.0, 5.0);

    case 'delay'
        % Replay / time-lag: position = pos(t - delta)
        delta = params.delta;
        for k = onset_idx:M
            t_lag = t(k) - delta;
            t_lag = max(t(1), t_lag);
            pos_attacked(k, :) = interp1(t, state9d(:,1:3), t_lag, 'linear', 'extrap');
        end
        % Recompute velocity / acceleration from attacked position
        dt_out = mean(diff(t));
        vel_attacked = gradient_nd(pos_attacked, dt_out);
        acc_attacked = gradient_nd(vel_attacked, dt_out);

    case 'takeover'
        % Smooth pull toward forged reference trajectory
        M_off  = params.M;
        alpha  = params.alpha;
        T_drift = params.M / (UAV_SPEED * alpha + eps);  % rough ramp duration
        % Reference = original position + growing offset (ramp)
        p_prime = pos_attacked;   % running state
        for k = onset_idx:M
            dt_k = t(k) - t_s;
            g    = min(1.0, dt_k / T_drift);
            p_ref_k = state9d(k, 1:2) + (M_off * g * d_hat)';
            if k == onset_idx
                p_prime(k, 1:2) = p_ref_k;
            else
                p_prime(k, 1:2) = p_prime(k-1, 1:2) + ...
                    alpha * (p_ref_k - p_prime(k-1, 1:2));
            end
        end
        pos_attacked = p_prime;
        dt_out = mean(diff(t));
        vel_attacked = gradient_nd(pos_attacked, dt_out);
        acc_attacked = gradient_nd(vel_attacked, dt_out);

    otherwise
        error('Unknown attack_type: %s', attack_type);
end

state9d_attacked = [pos_attacked, vel_attacked, acc_attacked];

attack_info.onset_time  = t_s;
attack_info.onset_idx   = onset_idx;
attack_info.attack_type = attack_type;
attack_info.params      = params;
attack_info.direction   = d_hat;
end

%% Helper: direction vector -----------------------------------------------
function d_hat = compute_direction(mode, state9d, onset_idx)
switch mode
    case 'random_xy'
        theta = 2 * pi * rand();
        d_hat = [cos(theta); sin(theta)];
    case 'along_track_xy'
        vel_2d = state9d(onset_idx, 4:5)';
        n = norm(vel_2d);
        if n < 1e-6, vel_2d = [1;0]; n = 1; end
        d_hat = vel_2d / n;
    case 'cross_track_xy'
        vel_2d = state9d(onset_idx, 4:5)';
        n = norm(vel_2d);
        if n < 1e-6, vel_2d = [1;0]; n = 1; end
        along = vel_2d / n;
        d_hat = [-along(2); along(1)];
    case 'fixed_east'
        d_hat = [1; 0];
    case 'fixed_north'
        d_hat = [0; 1];
    otherwise
        theta = 2 * pi * rand();
        d_hat = [cos(theta); sin(theta)];
end
end

%% Helper: transient velocity disturbance ---------------------------------
function [vel_out, acc_out] = add_velocity_transient(t, vel_in, acc_in, t_s, d_hat, v_max, tau, T_dur)
vel_out = vel_in;
for k = 1:length(t)
    if t(k) >= t_s && t(k) <= t_s + T_dur
        dt_k = t(k) - t_s;
        dv   = v_max * exp(-dt_k / tau);
        vel_out(k, 1:2) = vel_out(k, 1:2) + (dv * d_hat)';
    end
end
dt_out = mean(diff(t));
acc_out = gradient_nd(vel_out, dt_out);
end

%% Helper: central differences -------------------------------------------
function out = gradient_nd(X, dt)
out = zeros(size(X));
out(2:end-1, :) = (X(3:end,:) - X(1:end-2,:)) / (2*dt);
out(1,       :) = (X(2,:)     - X(1,:))        / dt;
out(end,     :) = (X(end,:)   - X(end-1,:))    / dt;
end
