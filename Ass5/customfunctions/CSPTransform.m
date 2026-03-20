function X = CSPTransform(X, V, num)
    
    first = V(:, num);
    second = V(:, [num, end]);

    new_v = cat(1, first, second);

    X = new_v' * X;
    
end