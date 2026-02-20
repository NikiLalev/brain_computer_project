function [accuracy_train, accuracy_test, w, b, line_coords] = custom_LDA(X_train, Y_train, X_test, Y_test)
% CUSTOM_LDA Implements Linear Discriminate Analysis for 2 classes
% Input 
%  X_train: 2xN (features x trials)
%  Y_train: 1xN (targets, should be 1 or -1)
%  X_test: 2xM (features x trials)
%  Y_test: 1xM (targets, should be 1 or -1)
% Output
%  accuracy_train: 0-1
%  accuracy_test: 0-1
%  w: prjection vector
%  b: bias
%  line_coords: struct with points that can be used for plotting the
%               decision line

% Seperate data by class
index_class_one = (Y_train == 1);
index_class_two = (Y_train == -1);

X_train_c1 = X_train(:, index_class_one);
X_train_c2 = X_train(:, index_class_two);

c1_train_size = sum(index_class_one);
c2_train_size = sum(index_class_two);
train_size = c1_train_size + c2_train_size;

% Compute mean and common cavariance
% 2x1 mean per feature
mean_class_one = mean(X_train_c1, 2); 
mean_class_two = mean(X_train_c2, 2);
% 2x2 covariance matrix for the 2 features of each class
cov_class_one = cov(X_train_c1(1,:), X_train_c1(2,:));
cov_class_two = cov(X_train_c2(1,:), X_train_c2(2,:));
% common covariance matrix - weighted average of individual class
% covariance matrices
combined_cov = ((c1_train_size - 1) * cov_class_one + (c2_train_size - 1) * cov_class_two) / (train_size - 2); 

% Compute projection vector
w = (combined_cov^-1) * (mean_class_one - mean_class_two);

% Compute bias
p1 = c1_train_size / (train_size);
p2 = c2_train_size / (train_size);
b = -0.5 * (mean_class_one') * (combined_cov^-1) * mean_class_one + 0.5 * (mean_class_two') * (combined_cov^-1) * mean_class_two  + log(p1/p2); 

% Result for test set
result = sign(w' * X_test + b);
correct = (result == Y_test);
accuracy_test = sum(correct) / length(Y_test);
fprintf('Test Set Accuracy: %.2f%%\n', accuracy_test * 100);

% Result for train set
result_train = sign(w' * X_train + b);
accuracy_train = sum(result_train == Y_train) / (train_size);
fprintf('Train Set Accuracy: %.2f%%\n', accuracy_train* 100);

% Compute decision line. We have 2 features x1 and x2 -> w1*x1 + w2*x2 + b.
% We need 2 points to define a line so just take max and min values of x1 and 
% find what are the corresponding x2 values via
% x2 = (-w1*x1 - b) / w2.
x1_min = min(X_train(1,:));
x1_max = max(X_train(1,:));

x2_one = (-w(1) * x1_min - b) / w(2);
x2_two = (-w(1) * x1_max - b) / w(2);

line_coords.x = [x1_min, x1_max];
line_coords.y = [x2_one, x2_two];