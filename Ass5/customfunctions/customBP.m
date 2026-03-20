function X = customBP(X)
    %It is expected that the first dimension of X is the time dimension.
    X = X .^ 2;
    X = sum(1, X);
    X = log10(X);
    X = real(X);    
end