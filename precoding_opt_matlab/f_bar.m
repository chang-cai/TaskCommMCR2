function fcn_bar = f_bar(lambda, g, Gamma, matrix_form)
%   Detailed explanation goes here

    fcn_bar = 0;
    if matrix_form == false
        for i = 1:size(Gamma, 1)
            fcn_i = norm(g(i))^2/( (Gamma(i, i) + lambda)^2 );
            fcn_bar = fcn_bar + fcn_i;
        end
    else
        for i = 1:size(Gamma, 1)
            fcn_i = abs(g(i))/( (Gamma(i, i) + lambda)^2 );
            fcn_bar = fcn_bar + fcn_i;
        end
    end
end