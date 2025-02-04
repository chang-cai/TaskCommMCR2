function H_slot = H_slot_rician_channel_gen(N_t_k, N_r, kappa)

AoA = deg2rad(360*rand(1));
AoD = deg2rad(360*rand(1));
H_LoS = steering_vec(sin(AoA), N_r) * steering_vec(sin(AoD), N_t_k)';
H_NLoS = 1/sqrt(2*N_r*N_t_k) * (randn(N_r, N_t_k) + 1j*randn(N_r, N_t_k));
% H_NLoS = 1/sqrt(2) * (randn(N_r, N_t_k) + 1j*randn(N_r, N_t_k));
H_slot = sqrt(kappa/(1+kappa))*H_LoS + sqrt(1/(1+kappa))*H_NLoS;

end