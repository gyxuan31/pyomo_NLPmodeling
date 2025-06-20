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
locux = randn(1, total_UE) * 10;
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
end

% NORMAL
rb_counts = randi([0, 5], 1, total_UE); % initial allocation
e_norm = zeros(total_UE, num_RB);

for i = 1:total_UE
    count = rb_counts(i);
    if count > 0
        selected_rbs = randperm(num_RB, count); 
        e_norm(i, selected_rbs) = 1;
    end
end

record_norm = [];
for t = 1:100
    data_rate = zeros(1, total_UE);
    for n = 1:total_UE
        for k = 1:num_RB
            if e_norm(n, k) == 1
                signal = P * distance(t, n, user_RU(n))^(-eta) * rayleigh_gain(n, k);

                interference = 0;
                for others = 1:total_UE
                    if others ~= n && e_norm(others, k) == 1
                        if user_RU(others) ~= user_RU(n)
                            interference = interference + ...
                                P * distance(t, others)^(-eta) * rayleigh_gain(others, k);
                        end
                    end
                end

                SINR = signal / (interference + sigmsqr);
                data_rate(n) = data_rate(n) + B * log(1 + SINR);
            end
        end
    end

    record_norm = [record_norm, sum(data_rate)];
end

figure;
plot(1:T, record_norm, 'LineWidth', 2);
xlabel('Time Step');
ylabel('Total Data Rate (bps)');
title('Total Data Rate over Time (Normal Allocation)');
grid on;
ylim([0, max(record_norm)*1.05]);  % 纵轴从0到最大值稍微放大一点

