function total_rate = compute_total_rate(e, predicted_len, total_UE, num_RB, pre_distance, rayleigh_gain, P, sigmsqr, eta, B, T, user_RU, num_RU)
    E = reshape(e, predicted_len, total_UE, num_RB);
    total_rate = 0;
    for t = 1:predicted_len
        data_rate = zeros(1, total_UE); % vector of cu
        for n = 1:total_UE
            for k = 1:num_RB
                signal = E(t, n, k) * P * pre_distance(t, n, user_RU(n))^(-eta) * rayleigh_gain(n, k);
                interference = 0;
                for others = 1:total_UE
                    interference = interference + E(t, others, k) * P * pre_distance(t, n, user_RU(others))^(-eta) * rayleigh_gain(n, k);
                end
                SINR = signal / (interference + sigmsqr);
                data_rate(n) = data_rate(n) + B * log(1 + SINR);
            end
        end
        total_rate = total_rate + sum(log(1+data_rate));
    end

    % for t = 1:predicted_len
    %     data_rate = zeros(1, total_UE); % vector of cu
    %     for n = 1:total_UE
    %         for k = 1:num_RB
    %             if E(t, n, k) >= 0.5
    %                 signal = P * pre_distance(t, n, user_RU(n))^(-eta) * rayleigh_gain(n, k);
    %                 interference = 0;
    %                 for m = 1:total_UE
    %                     for i = 1:num_RU
    %                         if m ~= n && E(t, m, k) >= 0.5 && user_RU(m) ~= user_RU(n)
    %                             interference = interference + P * pre_distance(t, n, user_RU(i))^(-eta) * rayleigh_gain(n, k);
    %                         end
    %                     end
    %                 end
    %                 SINR = signal / (interference + sigmsqr);
    %                 data_rate(n) = data_rate(n) + B * log(1 + SINR);
    %             end
    %         end
    %     end
    %     total_rate = total_rate + sum(log(1+data_rate));
    % end


end

