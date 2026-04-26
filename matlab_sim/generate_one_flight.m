function generate_one_flight(flight_id, split, out_dir)
%GENERATE_ONE_FLIGHT  Generate and save one flight CSV.
%
%   Deterministically generates flight based on flight_id only.
%   Called by mass_runner in a parfor loop.
%
%   split: 'train' | 'val' | 'test'

config;

% Seed is fully determined by flight_id to ensure reproducibility
base_seed = flight_id * 97 + 13;

% ---- pick mission profile (cycle through 4 types) ----
profile_type = mod(flight_id - 1, 4) + 1;
mission = generate_mission(profile_type, base_seed);

% ---- simulate clean flight ----
[t, state9d] = simulate_flight(mission, base_seed);

% ---- decide if this flight gets an attack ----
switch split
    case 'train'; attack_ratio = ATTACK_RATIO_TRAIN;
    case 'val';   attack_ratio = ATTACK_RATIO_VAL;
    case 'test';  attack_ratio = ATTACK_RATIO_TEST;
end

rng(base_seed + 500, 'twister');
has_attack = rand() < attack_ratio;

attack_types = {'step', 'drift', 'delay', 'takeover'};
directions   = {'random_xy', 'along_track_xy', 'cross_track_xy', 'fixed_east', 'fixed_north'};
step_M       = [5, 15, 30];
drift_M      = [10, 20, 30];
drift_T      = [5, 10, 20];
delay_delta  = [1.0, 3.0, 5.0];
takeover_M   = [10, 20, 30];
takeover_a   = [0.3, 0.5, 0.7];

label = zeros(length(t), 1);   % 0 = normal
onset_time = NaN;
attack_type_str = 'none';

if has_attack
    % Attack type cycles by flight_id for balanced distribution
    a_type = attack_types{mod(flight_id - 1, 4) + 1};
    dir_mode = directions{mod(floor((flight_id-1)/4), 5) + 1};

    params.onset_frac = ATTACK_ONSET_FRAC_MIN + ...
        (ATTACK_ONSET_FRAC_MAX - ATTACK_ONSET_FRAC_MIN) * rand();
    params.direction = dir_mode;

    switch a_type
        case 'step'
            params.M = step_M(mod(flight_id - 1, 3) + 1);
        case 'drift'
            params.M = drift_M(mod(flight_id - 1, 3) + 1);
            params.T_drift = drift_T(mod(floor((flight_id-1)/3), 3) + 1);
        case 'delay'
            params.delta = delay_delta(mod(flight_id - 1, 3) + 1);
        case 'takeover'
            params.M = takeover_M(mod(flight_id - 1, 3) + 1);
            params.alpha = takeover_a(mod(floor((flight_id-1)/3), 3) + 1);
    end

    [state9d, attack_info] = inject_attack(t, state9d, a_type, params, base_seed);
    onset_time = attack_info.onset_time;
    attack_type_str = a_type;
    label = double(t >= onset_time);
end

% ---- write CSV ----
T = array2table([t, state9d, label], ...
    'VariableNames', {'time','x','y','z','vx','vy','vz','ax','ay','az','label'});
T.attack_type = repmat({attack_type_str}, height(T), 1);
T.onset_time  = repmat(onset_time, height(T), 1);

fname = fullfile(out_dir, split, sprintf('flight_%06d.csv', flight_id));
writetable(T, fname);
end
