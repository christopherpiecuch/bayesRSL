%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize_output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code written by CGP 2017/02/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = nan(NN_thin,N);         % Spatial field of process linear trend
L = nan(NN_thin,N);         % Spatial field of observational biases
R = nan(NN_thin,1);         % AR(1) coefficient of the process
Y = nan(NN_thin,K,N);       % Process values
Y_0 = nan(NN_thin,N);       % Process initial conditions
MU = nan(NN_thin,1);        % Mean value of process linear trend
NU = nan(NN_thin,1);        % Mean value of observational biases
PHI = nan(NN_thin,1);       % Inverse range of process innovations
PI_2 = nan(NN_thin,1);      % Spatial variance of process linear trend
SIGMA_2 = nan(NN_thin,1);   % Sill of the process innovations
DELTA_2 = nan(NN_thin,1);   % Instrumental error variance 
TAU_2 = nan(NN_thin,1);     % Spatial variance in observational biases
