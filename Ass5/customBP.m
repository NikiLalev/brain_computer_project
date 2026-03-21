function X = customBP(X)
    % Input X: channels x samples x trials (4 x 512 x 100)
    % square signal
    X = X .^ 2;
    % sum over time / samples dim
    X = sum(X, 2);
    % take log
    X = log10(X);
    % remove imaginary components
    X = real(X);    
end