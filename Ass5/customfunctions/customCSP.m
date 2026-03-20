function [V, D] = customCSP(X, Y)
    nominator = X * X';
    denominator = trace(nominator);

    cov_all = nominator / denominator;

    labels = unique(Y);

    cov_1 = cov_all(Y == labels(1));
    cov_2 = cov_all(Y == labels(2));

    av_cov_1 = 1/size(cov_1, 1) * sum(cov_1, 2);
    av_cov_2 = 1/size(cov_2, 1) * sum(cov_2, 2);

    [V, D] = eig(av_cov_1, av_cov1 + av_cov_2, 'qz');

    [D, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
end