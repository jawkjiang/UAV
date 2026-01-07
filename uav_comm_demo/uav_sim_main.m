% uav_sim_main.m - 主仿真脚本

clear; clc; close all;
run('params.m');

% 初始化位置
uav_pos = uav_init_pos;
sinr_log = zeros(1, sim_time);

figure;
for t = 1:sim_time
    % UAV 到 BS 和 User 的距离
    d_uav_user = norm(uav_pos - user_pos);
    d_uav_bs = norm(uav_pos - bs_pos);
    d_user_bs = norm(user_pos - bs_pos);
    d_jam_bs = norm(jammer_pos - bs_pos);
    
    % 是否使用中继：判断 User–BS 距离是否超过设定阈值
    direct_sinr = 10*log10(Pt/d_user_bs^2 / (Pj/d_jam_bs^2 + N0));
    use_relay = direct_sinr < sinr_threshold;

    if use_relay
        % UAV 靠近中继点（user和bs的中点）
        target = (user_pos + bs_pos) / 2;
        direction = (target - uav_pos) / norm(target - uav_pos);
        uav_pos = uav_pos + direction * v_uav;
        
        % 两跳 SINR (最小信道质量决定中继性能)
        sinr_uav = 10*log10(Pt/d_uav_user^2 / N0);
        sinr_bs  = 10*log10(Pt/d_uav_bs^2 / (Pj/d_jam_bs^2 + N0));
        sinr_log(t) = min(sinr_uav, sinr_bs);
    else
        sinr_log(t) = direct_sinr;
    end
    
    % ---------- 可视化 ----------
    subplot(1,2,1); cla;
    plot(user_pos(1), user_pos(2), 'bo', 'MarkerSize',10, 'DisplayName','User');
    hold on;
    plot(jammer_pos(1), jammer_pos(2), 'rx', 'MarkerSize',10, 'DisplayName','Jammer');
    plot(bs_pos(1), bs_pos(2), 'ks', 'MarkerSize',10, 'DisplayName','Base Station');
    plot(uav_pos(1), uav_pos(2), 'g^', 'MarkerSize',10, 'DisplayName','UAV');
    legend(); axis equal; grid on;
    title(['t = ' num2str(t) ', SINR = ' num2str(sinr_log(t), '%.2f') ' dB']);
    xlim([-50 350]); ylim([-50 150]);

    subplot(1,2,2);
    plot(1:t, sinr_log(1:t), 'b', 'LineWidth', 2); grid on;
    xlabel('Time Step'); ylabel('SINR (dB)');
    title('SINR Over Time');

    pause(0.05); % 控制动画播放速度
end
