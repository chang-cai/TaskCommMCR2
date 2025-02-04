function [e] = steering_vec(x,M)
% steering vector function
m = (1:M)';
e = 1/sqrt(M) * exp(-1j*pi*(m-1)*x);
% e = exp(-1j*pi*(m-1)*x);
end

