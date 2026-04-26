%% mass_runner.m — Generate the full dataset in parallel
%
%  Run from within matlab_sim/ directory:
%    cd matlab_sim
%    mass_runner
%
%  Outputs: output/train/  output/val/  output/test/

config;

fprintf('=== UAV GPS Spoofing Simulation Dataset Generator ===\n');
fprintf('Target: %d train / %d val / %d test flights\n', N_TRAIN, N_VAL, N_TEST);

out_dir = OUT_DIR;
for s = {'train', 'val', 'test'}
    d = fullfile(out_dir, s{1});
    if ~exist(d, 'dir'); mkdir(d); end
end

%% Generate train flights (ID 1 : N_TRAIN)
fprintf('\n[1/3] Generating %d training flights...\n', N_TRAIN);
t0 = tic;
parfor fid = 1:N_TRAIN
    generate_one_flight(fid, 'train', out_dir);
end
fprintf('  Done in %.1f s\n', toc(t0));

%% Generate val flights (ID N_TRAIN+1 : N_TRAIN+N_VAL)
fprintf('\n[2/3] Generating %d validation flights...\n', N_VAL);
t0 = tic;
id_offset = N_TRAIN;
parfor fid = 1:N_VAL
    generate_one_flight(fid + id_offset, 'val', out_dir);
end
fprintf('  Done in %.1f s\n', toc(t0));

%% Generate test flights (ID N_TRAIN+N_VAL+1 : N_TRAIN+N_VAL+N_TEST)
fprintf('\n[3/3] Generating %d test flights...\n', N_TEST);
t0 = tic;
id_offset = N_TRAIN + N_VAL;
parfor fid = 1:N_TEST
    generate_one_flight(fid + id_offset, 'test', out_dir);
end
fprintf('  Done in %.1f s\n', toc(t0));

%% Print summary statistics
fprintf('\n=== Dataset summary ===\n');
for s = {'train', 'val', 'test'}
    files = dir(fullfile(out_dir, s{1}, 'flight_*.csv'));
    n_total = length(files);
    n_attack = 0;
    for f = files'
        T = readtable(fullfile(f.folder, f.name), 'VariableNamingRule','preserve');
        if any(T.label == 1); n_attack = n_attack + 1; end
    end
    fprintf('  %-8s: %d flights, %d attacked (%.0f%%)\n', ...
        s{1}, n_total, n_attack, 100*n_attack/n_total);
end
fprintf('\nDone. Dataset saved to %s\n', out_dir);
