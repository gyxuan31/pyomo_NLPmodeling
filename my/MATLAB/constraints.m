function [c, ceq] = constraints(e)
    global predicted_len total_UE num_RB 
    gamma = 3; % reused ratio
    num_setreq = 3;

    E = reshape(e, predicted_len, total_UE, num_RB);
    c = zeros(predicted_len, 1); % <=0
    for t = 1:predicted_len
        e_sum = 0;
        for n = 1:total_UE
            for k = 1:num_RB
                e_sum = e_sum + E(t,n,k);
            end
        end
        c(t) = e_sum - num_RB*gamma;
    end

    ceq = [];
end
