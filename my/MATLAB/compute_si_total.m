function obj = compute_si_total(x, m, n, P, R)
    D = reshape(x, [m, n]);
    s = R(:) .* sum(P .* D, 2);
    obj = -sum(s);  % ga 最小化
end

