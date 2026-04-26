%% config.m  — centralised simulation parameters
% All other scripts read from this file.

%% Dataset size
N_TRAIN   = 1800;   % training flights
N_VAL     = 600;    % validation flights
N_TEST    = 600;    % test flights (NEVER augmented)
ATTACK_RATIO_TRAIN = 0.40;  % fraction of flights with an attack
ATTACK_RATIO_VAL   = 0.40;
ATTACK_RATIO_TEST  = 0.40;

%% Sensor / flight parameters
FS        = 10;          % output sample rate [Hz] (decimated from sim rate)
SIM_FS    = 100;         % internal simulation rate [Hz]
FLIGHT_MIN_S = 30;       % minimum flight duration [s]
FLIGHT_MAX_S = 90;       % maximum flight duration [s]
ALT_M     = 50;          % fixed inspection altitude [m AGL]
UAV_SPEED = 5;           % nominal horizontal speed [m/s]

%% GPS sensor noise (realistic defaults, ISA atmosphere)
GPS_HPOS_SIGMA  = 1.5;   % horizontal position noise [m, 1-sigma]
GPS_VPOS_SIGMA  = 3.0;   % vertical position noise [m]
GPS_VEL_SIGMA   = 0.05;  % velocity noise [m/s]

%% IMU sensor noise
IMU_ACCEL_SIGMA = 0.02;  % accelerometer noise [m/s^2]
IMU_GYRO_SIGMA  = 0.001; % gyro noise [rad/s]

%% Attack onset range (fraction of flight duration from each end)
ATTACK_ONSET_FRAC_MIN = 0.15;
ATTACK_ONSET_FRAC_MAX = 0.60;

%% Output path (relative to matlab_sim/)
OUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'output');
