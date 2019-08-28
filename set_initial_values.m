%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v
% set_initial_values
% Define initial process and parameter values as described in section 3.4.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code written by CGP 2017/02/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mean parameters
mu=normrnd(HP.eta_tilde_mu,sqrt(HP.delta_tilde_mu_2));
nu=normrnd(HP.eta_tilde_nu,sqrt(HP.delta_tilde_nu_2));

% variance parameters
pi_2=min([1 1/randraw('gamma', [0,1/HP.nu_tilde_pi_2,HP.lambda_tilde_pi_2], [1,1])]); % use min to prevent needlessly large values
delta_2=min([1 1/randraw('gamma', [0,1/HP.nu_tilde_delta_2,HP.lambda_tilde_delta_2], [1,1])]); % use min to prevent needlessly large values
sigma_2=min([1 1/randraw('gamma', [0,1/HP.nu_tilde_sigma_2,HP.lambda_tilde_sigma_2], [1,1])]); % use min to prevent needlessly large values
tau_2=min([1 1/randraw('gamma', [0,1/HP.nu_tilde_tau_2,HP.lambda_tilde_tau_2], [1,1])]); % use min to prevent needlessly large values

% inverse length scale parameters
phi=exp(normrnd(HP.eta_tilde_phi,sqrt(HP.delta_tilde_phi_2)));

% spatial fields
b=(mvnrnd(mu*ones(N,1),pi_2*eye(N)))';        
l=(mvnrnd(nu*ones(N,1),tau_2*eye(N)))';        
        
% AR(1) parameter
r=HP.u_tilde_r+(HP.v_tilde_r-HP.u_tilde_r )*rand(1);

% process
y_0=zeros(N,1);
y=zeros(N,K);
