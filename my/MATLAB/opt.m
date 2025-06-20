total_UE = 15;
num_RB = 20;
num_RU = 3;
P = 0.3;
eta = 2;
B = 200e3;
sigmsqr = 10^((-173-30)/10);

dim = total_UE * num_RB;

distance = rand(total_UE, 1) * 10 + 5;
rayleigh_gain = abs(randn(total_UE, num_RB));
rb_counts = randi([1, 5], total_UE, 1);

fitnessFcn = @(e) -compute_total_rate(round(e), total_UE, num_RB, ...
    distance, rayleigh_gain, P, sigmsqr, eta, B, rb_counts, num_RU);
nvars = dim;

lb = zeros(1, dim);
ub = ones(1, dim);

options = optimoptions('particleswarm', ...
    'SwarmSize', 40, ...
    'MaxIterations', 100, ...
    'Display', 'iter', ...
    'UseParallel', false);

[x_opt, fval] = particleswarm(fitnessFcn, nvars, lb, ub, options);

e_opt = round(reshape(x_opt, total_UE, num_RB));
fprintf('Optimized Total Data Rate: %.2f Mbps\n', -fval/1e6);

imagesc(e_opt);
xlabel('RB Index');
ylabel('User Index');
title('Optimized RB Allocation (1 = Allocated)');
colorbar;





function total_rate = compute_total_rate(e, total_UE, num_RB, ...
    distance, rayleigh_gain, P, sigmsqr, eta, B, rb_counts, num_RU)

    e = reshape(e, total_UE, num_RB);
    total_rate = 0;

    for u = 1:total_UE
        if sum(e(u,:)) < rb_counts(u)
            total_rate = total_rate - 1e10;
            return;
        end
    end

    if sum(e, 'all') > num_RU * num_RB
        total_rate = total_rate - 1e10;
        return;
    end

    for u = 1:total_UE
        for k = 1:num_RB
            if e(u,k) == 1
                signal = P * distance(u)^(-eta) * rayleigh_gain(u,k);
                interference = 0;
                for up = 1:total_UE
                    if up ~= u && e(up,k) == 1
                        interference = interference + ...
                            P * distance(up)^(-eta) * rayleigh_gain(up,k);
                    end
                end
                SINR = signal / (interference + sigmsqr);
                total_rate = total_rate + B * log(1 + SINR);
            end
        end
    end
end
