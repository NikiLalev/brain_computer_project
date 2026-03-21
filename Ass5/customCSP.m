function [V, D] = customCSP(X, Y)
    % Inputs:
    % X Data matrix: channels x samples x trials
    % Y Labels vec: trials x 1

    % can't multiply 3D matrices. We need 2 things. 1) Per trial covariances
    % matrices and 2) per class avg. covariance matrices
    [num_channels, num_samples, num_trials] = size(X);
    
    % We don't need to store individual covs, instead can store the sum
    % which is required to get the avg
    
    % this will be 15 x 15 in our case
    cov_sum_one = zeros(num_channels, num_channels);
    cov_sum_two = zeros(num_channels, num_channels);
    % get the labels from Y
    labels = unique(Y);

    % computing trials per class
    for i = 1:num_trials
        % 15 x 512
        trial_data = X(:,:, i);
        % spatial normalized cov per trial
        numerator = trial_data * trial_data';
        denominator = trace(numerator);
        cov_trial = numerator / denominator;
        % add the cov to the covariance sum for the correct class
        if Y(i) == labels(1)
            cov_sum_one = cov_sum_one + cov_trial; 
        elseif Y(i) == labels(2)
            cov_sum_two = cov_sum_two + cov_trial; 
        end
    end

    class_counts = groupcounts(Y);
    % Compute avg cov matrices for each class
    avg_cov_one = cov_sum_one / class_counts(1);
    avg_cov_two = cov_sum_two / class_counts(2);

    % solve eigenvalue problem
    [V, D] = eig(avg_cov_one, avg_cov_one + avg_cov_two, 'qz');
    % sort in descending order using the eigenvalues on the diag of D
    [D, sort_idx] = sort(diag(D), 'descend');
    V = V(:, sort_idx);
end