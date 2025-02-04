function lambda = bisection_search_bar(lambda_lb, lambda_ub, P, epsilon, g, Gamma, matrix_form)
%   Detailed explanation goes here

lb = lambda_lb;
ub = lambda_ub;
lambda = lambda_ub;

while ub - lb >= epsilon
    lambda = (lb + ub)/2;
    if f_bar(lambda, g, Gamma, matrix_form) <= P
        ub = lambda;
    else
        lb = lambda;
    end
end

end