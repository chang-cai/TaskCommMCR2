function [f, log_f] = gaussian_pdf(x, mean, covariance, epsilon)
% multivariate gaussian pdf

n = length(x);
coef_1 = 1/sqrt((2*pi)^n);
% det(covariance+epsilon*eye(n))
coef_2 = 1/sqrt(det(covariance+epsilon*eye(n)));
coef = coef_1 * coef_2;
coef_3 = -0.5*(x-mean)'*inv(covariance+epsilon*eye(n))*(x-mean);
expo = exp(coef_3);
f = coef*expo;

%%
log_f = log(coef)+coef_3;

end