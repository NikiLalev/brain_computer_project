function y = custom_dft(x)
%CUSTOM_DFT Summary of this function goes here
%   Detailed explanation goes here
  
    a_k = zeros(size(x));
    N = length(x);
    for k = 1:N
        a_k(k) = sum(x .* exp(-2*pi*1i*(k-1)*(0:N-1)/N));
    end

    y=a_k;
end