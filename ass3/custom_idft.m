function x = custom_idft(y)
%CUSTOM_IDFT Summary of this function goes here
%   Detailed explanation goes here


    a_k = zeros(size(y));
    N = length(y);
    for k = 1:N
        a_k(k) = 1/N * sum(y .* exp(2*pi*1i*(k-1)*(0:N-1)/N));
    end
    
    x=a_k;

end