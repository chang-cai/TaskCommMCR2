clc
clear

load("MCR2_ModelNet10_statistics_complex_24.mat", "feature_cplx", "label",...
    "C_xx_real", "mean_real", "C_xx", "CLS_mean", "CLS_cov", "CLS_rlt");
load("MCR2_ModelNet10_test_feature_label_complex_24.mat", "p", "test_feature_cplx", "test_label");
test_feature = test_feature_cplx.';


B = 10*1e3;  % 10 kHz
K = 3;  % num. of devices
N_t_k = 4;  % num. of transmit antennas
N_t = N_t_k * K;
N_r = 8;  % num. of receive antennas
T = 1;  % num. of time slots

Dataset = size(test_feature, 2);
% NMSE_factor = norm(test_feature, "fro")^2/Dataset;
C = C_xx;  % covariance matrix
L = 10;  % num. of classes
D = 2*size(test_feature, 1);

delta_0_square = 1e-20*B;  % noise power (W)  -170 dBm/Hz density
delta_0 = sqrt(delta_0_square);  % noise
P_k_dBm = 0;  % dBm
P_k = 1e-3*10.^(P_k_dBm./10);  % W
P = P_k * K;

d = 240;  % m
PL = 32.6 + 36.7*log10(d);  % dB
PL = 10^(-PL/10);
delta_0 = delta_0/sqrt(PL);  % normalize path loss into the noise power

eps = 1e-3;

% channel generation
kappa = 1;  % Rician factor
H = [];
H_k_all = zeros(T*N_r, T*N_t_k, K);
H_k_t_all = zeros(N_r, N_t_k, K);
for k = 1:K
    H_k_t = H_slot_rician_channel_gen(N_t_k, N_r, kappa);
    H_k_t_all(:, :, k) = H_k_t;
%             svd(H_k_t)
    H_k = kron(eye(T), H_k_t);
    H_k_all(:, :, k) = H_k;
    H = [H, H_k];
end

%% MCR2 Precoder
alpha = T*N_r / eps^2;
C_sr = sqrtm(C);

% initialization of V
V_init_k = rand(T*N_t_k, D/(2*K)) + 1j*rand(T*N_t_k, D/(2*K));
V_init = kron(eye(K), V_init_k);
V_init = sqrt(P) * V_init ./ sqrt(trace(V_init*C*V_init'));
V = V_init;

Ite = 1000;
mcr2_obj_last = 500;
mcr2_all = zeros(1, Ite);
for ii = 1:Ite
    ii;
    % U update
    U = (H*V*C*V'*H'+(1/alpha + delta_0^2)*eye(N_r*T))\(H*V*C_sr);

    % W update
    E_0 = eye(D/2)-U'*H*V*C_sr;
    E_0 = (E_0*E_0') + (1/alpha + delta_0^2)*(U'*U);
    W_0 = inv(E_0);
    W_j_all = zeros(N_r*T, N_r*T, L);
    for j = 1:L
        C_j = CLS_cov(:, :, j);
        W_j_all(:, :, j) = inv((1+alpha*delta_0^2)*eye(N_r*T) + alpha*H*V*C_j*V'*H');
    end
    
    % V update
    for k = 1:K
        H_k = H_k_all(:, :, k);
        C_sr_k = C_sr(((k-1)*D/(2*K)+1):k*D/(2*K), :);
        C_k = C(((k-1)*D/(2*K)+1):k*D/(2*K), :);
        C_kk = C(((k-1)*D/(2*K)+1):k*D/(2*K), ((k-1)*D/(2*K)+1):k*D/(2*K));
        V_k = V(((k-1)*T*N_t_k+1):k*T*N_t_k, ...
            ((k-1)*D/(2*K)+1):k*D/(2*K));

        sum_term1_T_k = H*V*C_k' - H_k*V_k*C_kk;
        sum_term2_T_k = 0;  % zeros(2*T*N_t_k, D/K);
        sum_term_M_k = 0;
%                 I_kron_H_bar_k = kron(eye(D/(2*K)), H_k);
        for j = 1:L
            W_j = W_j_all(:, :, j);

            C_j_k = CLS_cov(((k-1)*D/(2*K)+1):k*D/(2*K), :, j);
            C_j_kk = CLS_cov(((k-1)*D/(2*K)+1):k*D/(2*K), ((k-1)*D/(2*K)+1):k*D/(2*K), j);
            sum_term2_j_T_k = H*V*C_j_k' - H_k*V_k*C_j_kk;
            sum_term2_T_k = sum_term2_T_k + ...
                alpha*p(j)*H_k'*W_j*sum_term2_j_T_k;

%                     sum_term_M_k = sum_term_M_k + ...
%                         alpha*p(j)*I_kron_H_bar_k'*kron(C_j_kk, W_j)*I_kron_H_bar_k;
            sum_term_M_k = sum_term_M_k + ...
                alpha*p(j)*kron(C_j_kk.', H_k'*W_j*H_k);
        end
        T_k = H_k'*U*W_0*C_sr_k' - ...
            H_k'*U*W_0*U'*sum_term1_T_k - sum_term2_T_k;
%                 I_kron_U_H_bar_k = kron(eye(D/K), U'*H_k);
        M_k = kron(C_kk.', H_k'*U*W_0*U'*H_k) + ...
            sum_term_M_k;

        [Q_kk, D_kk] = svd(C_kk.');
        Q_kk_kron_I = kron(Q_kk, eye(N_t_k*T));
        D_kk_kron_I = kron(D_kk, eye(N_t_k*T));
        D_kk_kron_I_msr = diag( 1./sqrt(diag(D_kk_kron_I)) );
        C_kk_kron_I_msr = Q_kk_kron_I * D_kk_kron_I_msr * Q_kk_kron_I';

        t_k = C_kk_kron_I_msr*vec(T_k);
        M_k = C_kk_kron_I_msr*M_k*C_kk_kron_I_msr;


        [U_M_k, Gamma_k] = svd(M_k);
        g_k = U_M_k' * t_k;
        
        lambda_lb = 0;
        lambda_ub = sqrt(real(g_k'*g_k)/(P_k*T));
        
        fcn_bar_0 = f_bar(0, g_k, Gamma_k, false);
        epsilon_bi_search = 1e-6;
        if fcn_bar_0 <= P_k
            lambda = 0;
        else
           lambda = bisection_search_bar(lambda_lb, lambda_ub, T*P_k, epsilon_bi_search, g_k, Gamma_k, false);
        end
        
        if lambda < 1e-20
            v_k = pinv( M_k + lambda*eye(N_t_k*T*D/(2*K)) ) * t_k;
        else
            v_k = ( M_k + lambda*eye(N_t_k*T*D/(2*K)) ) \ t_k;
        end
        
        v_k = C_kk_kron_I_msr * v_k;
        
        % retrive V_k from v_k
        V_k = zeros(T*N_t_k, D/(2*K));
        for i = 1:D/(2*K)
            V_k(:, i) = v_k(T*N_t_k*(i-1)+1: T*N_t_k*i);
        end
        V((T*N_t_k*(k-1)+1):T*N_t_k*k, ((k-1)*D/(2*K)+1):k*D/(2*K)) = V_k;
    end
    [mcr2, ~, ~] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p);
    mcr2_all(ii) = mcr2;

    % terminate criterion
    if abs((mcr2 - mcr2_obj_last)/mcr2_obj_last) < 1e-5
        break;
    end
    mcr2_obj_last = mcr2;
end
figure;
plot(mcr2_all(1:ii), 'r-', LineWidth=1.6);
[mcr2, ~, ~] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p);

%% MAP receiver and LMMSE estimator
epsilon = 1e-6; % trick
acc_mcr2 = 0;
mse_by_mcr2 = 0;
parfor j = 1:Dataset
    z_cplx = test_feature(:, j);
    % add noise
    n_cplx = delta_0 * sqrt(1/2) * (randn(N_r*T, 1) + 1j*randn(N_r*T, 1));
    C_n_cplx = delta_0^2 * eye(N_r*T);
    y_cplx = H*V*z_cplx + n_cplx;

    %% Recover signals from complex to real
    z_real = [real(z_cplx); imag(z_cplx)];
    y_real = [real(y_cplx); imag(y_cplx)];
    HV_real = [real(H*V), -imag(H*V);...
        imag(H*V), real(H*V)];
    C_n_real = 1/2 * [real(C_n_cplx), zeros(N_r*T, N_r*T);
        zeros(N_r*T, N_r*T), real(C_n_cplx)];

    %%
    f = zeros(1, L);
    log_f = zeros(1, L); % numerical issue
    weighted_sum_element = zeros(1, L);
    for i = 1:L
        mu_i_cplx = CLS_mean(:, i);
        C_i = CLS_cov(:, :, i);
        R_i = CLS_rlt(:, :, i);
        
        mu_i = [real(mu_i_cplx); imag(mu_i_cplx)];
        Cov_i = 1/2 * [real(C_i + R_i), imag(-C_i + R_i);...
            imag(C_i + R_i), real(C_i - R_i)];

        [f(i), log_f(i)] = gaussian_pdf(y_real, HV_real*mu_i,...
            HV_real*Cov_i*HV_real' + C_n_real, epsilon);
        weighted_sum_element(i) = p(i) * f(i);
    end
    weighted_sum = sum(weighted_sum_element);
    
    % GM receiver
%             [~, pos] = max(log_f);
    [~, pos] = max(weighted_sum_element);
    if (pos-1)==test_label(j)
        acc_mcr2 = acc_mcr2 +1;
    end
    
    % LMMSE estimate
    z_lmmse_estimate = mean_real + C_xx_real*HV_real'*...
        inv(HV_real*C_xx_real*HV_real'+C_n_real+epsilon*eye(2*N_r*T))*...
        (y_real-HV_real*mean_real);
    mse_by_mcr2 = mse_by_mcr2 + norm(z_lmmse_estimate-z_real, 2)^2;

end
acc_mcr2 = acc_mcr2/Dataset
mse_by_mcr2 = mse_by_mcr2/Dataset


%% LMMSE Precoder
alpha = T*N_r / eps^2;

% initialization of V
V_init_k = rand(T*N_t_k, D/(2*K)) + 1j*rand(T*N_t_k, D/(2*K));
V_init = kron(eye(K), V_init_k);
V_init = sqrt(P) * V_init ./ sqrt(trace(V_init*C*V_init'));
V = V_init;

Ite = 1000;
mse_obj_last = 500;
mse_all = zeros(1, Ite);
for ii = 1:Ite
    ii;
    % R update
    R = (H*V*C*V'*H'+(delta_0^2)*eye(N_r*T))\(H*V*C);
    
    % V update
    for k = 1:K
        H_k = H_k_all(:, :, k);
        C_k = C(((k-1)*D/(2*K)+1):k*D/(2*K), :);
        C_kk = C(((k-1)*D/(2*K)+1):k*D/(2*K), ((k-1)*D/(2*K)+1):k*D/(2*K));
        V_k = V(((k-1)*T*N_t_k+1):k*T*N_t_k, ...
            ((k-1)*D/(2*K)+1):k*D/(2*K));

        sum_term_T_k = H*V*C_k' - H_k*V_k*C_kk;
        J_k = C_k - sum_term_T_k'*R;
        
        Q_k = H_k'*R*J_k'/C_kk;
        M_k = H_k'*R*R'*H_k;

        [U_M_k, Gamma_k] = svd(M_k);
        G_k = U_M_k'*Q_k*C_kk*Q_k'*U_M_k;
        
        lambda_lb = 0;
        lambda_ub = sqrt(real(sum(diag(G_k)))/(T*P_k));
        
        fcn_bar_0 = f_bar(0, diag(G_k), Gamma_k, true);
        epsilon_bi_search = 1e-6;
        if fcn_bar_0 <= T*P_k
            lambda = 0;
        else
            lambda = bisection_search_bar(lambda_lb, lambda_ub, T*P_k, epsilon_bi_search, diag(G_k), Gamma_k, true);
        end
        V_k = (M_k + lambda*eye(size(M_k)))\Q_k;

        V((T*N_t_k*(k-1)+1):T*N_t_k*k, ((k-1)*D/(2*K)+1):k*D/(2*K)) = V_k;
    end
    matrix_inv = inv(H*V*C*V'*H'+delta_0^2*eye(N_r*T));
    mse_obj = real( trace(C) - trace(C*V'*H'*matrix_inv*H*V*C) );
    mse_all(ii) = mse_obj;

    % terminate criterion
    if abs((mse_obj - mse_obj_last)/mse_obj_last) < 1e-5
        break;
    end
    mse_obj_last = mse_obj;
end
figure;
plot(mse_all(1:ii), 'r-', LineWidth=1.6)
[mcr2, ~, ~] = MCR2_obj_cplx(V, H, C, CLS_cov, alpha, delta_0, p);

%% MAP receiver and LMMSE estimator
epsilon = 1e-6; % trick
acc_lmmse = 0;
mse_by_lmmse = 0;
parfor j = 1:Dataset
    z_cplx = test_feature(:, j);
    % add noise
    n_cplx = delta_0 * sqrt(1/2) * (randn(N_r*T, 1) + 1j*randn(N_r*T, 1));
    C_n_cplx = delta_0^2 * eye(N_r*T);
    y_cplx = H*V*z_cplx + n_cplx;

    %% Recover signals from complex to real
    z_real = [real(z_cplx); imag(z_cplx)];
    y_real = [real(y_cplx); imag(y_cplx)];
    HV_real = [real(H*V), -imag(H*V);...
        imag(H*V), real(H*V)];
    C_n_real = 1/2 * [real(C_n_cplx), zeros(N_r*T, N_r*T);
        zeros(N_r*T, N_r*T), real(C_n_cplx)];

    %%
    f = zeros(1, L);
    log_f = zeros(1, L); % numerical issue
    weighted_sum_element = zeros(1, L);
    for i = 1:L
        mu_i_cplx = CLS_mean(:, i);
        C_i = CLS_cov(:, :, i);
        R_i = CLS_rlt(:, :, i);
        
        mu_i = [real(mu_i_cplx); imag(mu_i_cplx)];
        Cov_i = 1/2 * [real(C_i + R_i), imag(-C_i + R_i);...
            imag(C_i + R_i), real(C_i - R_i)];

        [f(i), log_f(i)] = gaussian_pdf(y_real, HV_real*mu_i,...
            HV_real*Cov_i*HV_real' + C_n_real, epsilon);
        weighted_sum_element(i) = p(i) * f(i);
    end
    weighted_sum = sum(weighted_sum_element);
    
    % GM receiver
%             [~, pos] = max(log_f);
    [~, pos] = max(weighted_sum_element);
    if (pos-1)==test_label(j)
        acc_lmmse = acc_lmmse +1;
    end

    % LMMSE estimate
    z_lmmse_estimate = mean_real + C_xx_real*HV_real'*...
        inv(HV_real*C_xx_real*HV_real'+C_n_real+epsilon*eye(2*N_r*T))*...
        (y_real-HV_real*mean_real);
    mse_by_lmmse = mse_by_lmmse + norm(z_lmmse_estimate-z_real, 2)^2;

end
acc_lmmse = acc_lmmse/Dataset
mse_by_lmmse = mse_by_lmmse/Dataset
