global predicted_len total_UE num_RB
T = 100;
num_RU = 3;
UERU = 5;
total_UE = UERU * num_RU;
user_RU = zeros(1, total_UE); % RU index for every user

num_RB = 20;
B = 200e3;
P = 0.3;
sigmsqr = 10^((-173 - 30)/10);
eta = 2;
predicted_len = 5;

rayleigh_gain = ones(total_UE, num_RB);

% Location
locrux = [-5, 0, 5];
locruy = [-5, 0, 5];
locux = randn(1, total_UE) * 10; % initial user location
locuy = randn(1, total_UE) * 10;

trajectory_x = zeros(T, total_UE); % shape(T, total_UE)
trajectory_y = zeros(T, total_UE);
trajectory_x(1,:) = locux;
trajectory_y(1,:) = locuy;

for t = 1:T
    for i = 1:total_UE
        move_x = -1 + 2 * rand();  % np.random.uniform(-1, 1)
        move_y = -1 + 2 * rand();
        trajectory_x(t, i) = trajectory_x(max(t-1,1), i) + move_x;
        trajectory_y(t, i) = trajectory_y(max(t-1,1), i) + move_y;
    end
end

% Plot the user movement
% figure;
% plot(trajectory_x(:,1), trajectory_y(:,1), '-o', 'LineWidth', 2);
% xlabel('X');
% ylabel('Y');
% title('Trajectory of User 1');
% grid on;
% axis equal;

distance = zeros(T, total_UE, num_RU);
for t = 1:T
    for i = 1:total_UE
        temp = zeros(1, num_RU);
        for j = 1:num_RU
            dis = sqrt((trajectory_x(t,i) - locrux(j))^2 + (trajectory_y(t,i) - locruy(j))^2);
            temp(j) = dis;
            distance(t,i,j) = dis;
        end
        [~, user_RU(i)] = min(temp);
    end
    % Plot the connection of user and RU
    % figure;
    % hold on;
    % for i = 1:total_UE
    %     ru_idx = user_RU(i);
    %     plot([trajectory_x(t, i), locrux(ru_idx)], ...
    %          [trajectory_y(t, i), locruy(ru_idx)], ...
    %          'k-', 'LineWidth', 0.8);
    % end
    % scatter(trajectory_x(t, :), trajectory_y(t, :), 20, 'b', 'filled');
    % scatter(locrux, locruy, 20, 'r', 'filled');
    % xlabel('X');
    % ylabel('Y');
    % axis equal;
    % grid on;
end

% NORMAL - randomly generate e
rb_counts = randi([0, 5], 1, total_UE); % initial allocation
e_norm = zeros(total_UE, num_RB);
for i = 1:total_UE
    count = rb_counts(i);
    if count > 0
        selected_rbs = randperm(num_RB, count); 
        e_norm(i, selected_rbs) = 1;
    end
end

% OP - init
nvars = predicted_len * total_UE * num_RB;

lb = [];
ub = [];
for i=1:nvars
    lb = [lb,-1e-6];
    ub = [ub, 1+1e-6];
end

record_norm = [];
record_op = [];

for t = 1:10
    fprintf('t =  %d\n', t);
    data_rate_norm = zeros(1, total_UE);
    data_rate_op = zeros(1, total_UE);

    % NORMAL
    for n = 1:total_UE
        for k = 1:num_RB
            if e_norm(n, k) == 1
                signal = P * distance(t, n, user_RU(n))^(-eta) * rayleigh_gain(n, k);
                interference = 0;
                for others = 1:total_UE
                    for i = 1:num_RU
                        if others ~= n && e_norm(others, k) == 1 && user_RU(others) ~= user_RU(n)
                            interference = interference + ...
                                P * distance(t, n, user_RU(i))^(-eta) * rayleigh_gain(n, k);
                        end
                    end
                end
                SINR = signal / (interference + sigmsqr);
                data_rate_norm(n) = data_rate_norm(n) + B * log(1 + SINR);
            end
        end
    end
    record_norm = [record_norm, sum(log(1+data_rate_norm))];

    % OP
    pre_distance = distance(t:t+predicted_len-1, :, :);
    % fitnessFcn = @(e) -compute_total_rate(round(e), predicted_len, total_UE, num_RB, pre_distance, rayleigh_gain, P, sigmsqr, eta, B, T, user_RU, num_RU);
    % options = optimoptions('ga', 'PopulationSize', 40, 'MaxGenerations', 50, 'Display', 'iter');
    % [e_opt, fval] = ga(fitnessFcn, nvars, [], [], [], [], lb, ub, @constraints, options);
    fmin = @(e) -compute_total_rate(e, predicted_len, total_UE, num_RB, pre_distance, rayleigh_gain, P, sigmsqr, eta, B, T, user_RU, num_RU);
    options = optimoptions('fmincon', 'MaxIterations', 100, 'Display', 'iter-detailed', 'MaxFunctionEvaluations', 1e5);
    [e_opt, fval] = fmincon(fmin, repmat(e_norm, predicted_len, 1, 1), [], [], [], [], lb, ub, @constraints, options);


    for i = 1: nvars
        if e_opt(i) >= 0.5
            e_opt(i) = 1;
        else
            e_opt(i) = 0;
        end
    end
    e_opt = reshape(e_opt, predicted_len, total_UE, num_RB);
    
    for n = 1:total_UE
        for k = 1:num_RB
            if e_opt(1, n, k) >= 0.5
                signal = P * distance(t, n, user_RU(n))^(-eta) * rayleigh_gain(n, k);
                interference = 0;
                for others = 1:total_UE
                    for i = 1:num_RU
                        if others ~= n && e_opt(1, others, k) >= 0.5 && user_RU(others) ~= user_RU(n)
                            interference = interference + ...
                                P * distance(t, n, user_RU(i))^(-eta) * rayleigh_gain(n, k);
                        end
                    end
                end
                SINR = signal / (interference + sigmsqr);
                data_rate_op(n) = data_rate_op(n) + B * log(1 + SINR);
            end
        end
    end
    record_op = [record_op, sum(log(1+data_rate_op))];


    fprintf('Normal data rate: %.2f\n', record_norm(t));
    fprintf('Optmed data rate: %.2f\n', record_op(t));
    
    % subplot(1,2,1);
    % imagesc(squeeze(e_norm));
    % xlabel('RB Index');
    % ylabel('User Index');
    % title('Optimized RB Allocation (1 = Allocated)');
    % 
    % subplot(1,2,2);
    % imagesc(squeeze(e_opt(t, :, :)));
    % xlabel('RB Index');
    % ylabel('User Index');
    % title('Optimized RB Allocation (1 = Allocated)');
    % colorbar;


end

figure;
hold on;
t_len = length(record_norm);
plot(1:t_len, record_norm, 'LineWidth', 2, 'Color', 'r');
plot(1:t_len, record_op, 'LineWidth', 2, 'Color', 'b');

xlabel('Time Step');
ylabel('Total Data Rate (bps)');
title('Total Data Rate Comparison Over Time');
legend('Normal Allocation', 'Optimized Allocation');
grid on;
% ylim([0, max(record_op)*1.05]);



