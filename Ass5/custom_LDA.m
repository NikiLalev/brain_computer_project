function [w,b] = custom_LDA(X_train,y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

X1_train = X_train(y==1,:);
X2_train = X_train(y==2,:);

n1 = length(X1_train);
n2 = length(X2_train);

m1 = mean(X1_train);
m2 = mean(X2_train);

priori1 = n1/(n1+n2);
priori2 = n2/(n1+n2);

Sigma_c = (n1*cov(X1_train) + n2*cov(X2_train))./(n1 + n2);
w = inv(Sigma_c)*(m1' - m2');
b = -0.5*(m1*inv(Sigma_c)*(m1')) + 0.5*(m2*inv(Sigma_c)*(m2')) + log(priori1/priori2);


end