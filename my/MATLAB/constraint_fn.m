function [c, ceq] = constraint_fn(x, m, n, P, R, T, c_min, B)
    D = reshape(x, [m, n]);
    s = R(:) .* sum(P .* D, 2);

    c1 = sum(D,1)' - T(:);                   % n×1
    c2 = c_min(:) - sum(D,2);               % m×1
    c3 = s(:) - B(:);                       % m×1
    c = [c1; c2; c3];                       % (n + 2m) × 1

    ceq = [];
end
