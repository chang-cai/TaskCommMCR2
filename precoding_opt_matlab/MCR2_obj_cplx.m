function [mcr2, mcr2_1, mcr2_2] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p)

Dim = size(H, 1);
beta = 1 + alpha*delta_0^2;

mcr2_1 = log( det( beta*eye(Dim) + alpha*H*V*C*V'*H' ) );

mcr2_2 = 0;
class = size(p, 2);
for j = 1:class
    C_j = CLS_cov(:, :, j);
    mcr2_2 = mcr2_2 + p(j) * ...
        log( det( beta*eye(Dim) + alpha*H*V*C_j*V'*H' ) );
end
% mcr2 = 0.5 * real(mcr2_1 - mcr2_2);
% 
% mcr2_1 = 0.5 * real(mcr2_1);
% mcr2_2 = 0.5 * real(mcr2_2);
mcr2 = real(mcr2_1 - mcr2_2);

mcr2_1 = real(mcr2_1);
mcr2_2 = real(mcr2_2);
end