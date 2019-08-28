%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% delete_burn_in
% Delete warm-up iterations from solutions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code written by CGP 2017/02/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MU(1:NN_burn_thin)=[];
NU(1:NN_burn_thin)=[];
PI_2(1:NN_burn_thin)=[];
DELTA_2(1:NN_burn_thin)=[];
SIGMA_2(1:NN_burn_thin)=[];
TAU_2(1:NN_burn_thin)=[];
PHI(1:NN_burn_thin)=[];
B(1:NN_burn_thin,:)=[];
L(1:NN_burn_thin,:)=[];
R(1:NN_burn_thin)=[];
Y_0(1:NN_burn_thin,:)=[];
Y(1:NN_burn_thin,:,:)=[];