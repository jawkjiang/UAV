function mission = generate_mission(profile_type, rng_state)
%GENERATE_MISSION  Generate a UAV waypoint mission for one flight.
%
%   mission = generate_mission(profile_type, rng_state)
%
%   profile_type : 1=power_line_scan  2=rect_area_scan
%                  3=tower_orbit       4=long_transit
%   rng_state    : RNG seed (integer) for reproducibility
%
%   Returns mission struct with fields:
%     waypoints   [N×3] ENU coordinates [m]
%     duration    scalar total flight time [s]
%     profile     string description

rng(rng_state, 'twister');
config;   % load shared parameters

switch profile_type
    case 1
        % --- Power-line scan: back-and-forth along a line
        n_passes = randi([2, 6]);
        line_len = 50 + 80 * rand();          % 50-130 m per pass
        spacing  = 5 + 10 * rand();           % 5-15 m between passes
        wp = zeros(2*n_passes, 3);
        for k = 1:n_passes
            y_off = (k-1) * spacing;
            wp(2*k-1, :) = [0,        y_off, ALT_M];
            wp(2*k,   :) = [line_len, y_off, ALT_M];
        end
        profile = 'power_line_scan';

    case 2
        % --- Rectangular area scan (lawnmower)
        width  = 30 + 70 * rand();   % 30-100 m
        height = 30 + 70 * rand();
        n_rows = randi([3, 8]);
        row_spacing = height / (n_rows - 1 + eps);
        wp = zeros(2*n_rows, 3);
        for k = 1:n_rows
            y_off = (k-1) * row_spacing;
            if mod(k,2)==1
                wp(2*k-1,:) = [0,     y_off, ALT_M];
                wp(2*k,  :) = [width, y_off, ALT_M];
            else
                wp(2*k-1,:) = [width, y_off, ALT_M];
                wp(2*k,  :) = [0,     y_off, ALT_M];
            end
        end
        profile = 'rect_area_scan';

    case 3
        % --- Tower orbit (circular orbit approximated by polygon)
        radius   = 10 + 30 * rand();    % 10-40 m radius
        n_points = 8 + randi([0, 8]);   % 8-16 vertices
        angles   = linspace(0, 2*pi*(1 - 1/n_points), n_points);
        n_laps   = randi([1, 3]);
        single_lap = [radius*cos(angles)', radius*sin(angles)', repmat(ALT_M, n_points, 1)];
        wp = repmat(single_lap, n_laps, 1);
        profile = 'tower_orbit';

    case 4
        % --- Long transit: straight or slightly curved A-to-B
        dist   = 100 + 200 * rand();    % 100-300 m
        n_kink = randi([0, 3]);         % optional intermediate waypoints
        if n_kink == 0
            wp = [0, 0, ALT_M; dist, 0, ALT_M];
        else
            xs = sort(rand(n_kink, 1) * dist);
            ys = (rand(n_kink, 1) - 0.5) * 20;  % ±10 m lateral deviation
            mid_wps = [xs, ys, repmat(ALT_M, n_kink, 1)];
            wp = [[0, 0, ALT_M]; mid_wps; [dist, 0, ALT_M]];
        end
        profile = 'long_transit';

    otherwise
        error('Unknown profile_type: %d', profile_type);
end

% Compute approximate duration based on total path length
total_dist = 0;
for k = 2:size(wp,1)
    total_dist = total_dist + norm(wp(k,:) - wp(k-1,:));
end
approx_dur = total_dist / UAV_SPEED;
% Clamp to [FLIGHT_MIN_S, FLIGHT_MAX_S]
approx_dur = max(FLIGHT_MIN_S, min(FLIGHT_MAX_S, approx_dur));

mission.waypoints = wp;
mission.duration  = approx_dur;
mission.profile   = profile;
mission.rng_state = rng_state;
end
