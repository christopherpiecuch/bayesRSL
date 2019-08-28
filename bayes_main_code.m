%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function bayes_main_code(save_tag,NN_burn,NN_post,thin_period)
%   DESCRIPTION: Main driver code. Input are:
%       * save_tag: prefix to be appended to file name (see line 317).
%       * NN_burn: Number of burn-in iterations
%           (NN_burn = 100000 was used in the paper)
%       * NN_post: Number of post-burn-in iterations
%           (NN_post = 100000 was used in the paper)
%       * thin_period: Number of iterations to thin by
%           (thin_period = 100 was used in the paper)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code written by CGP 2017/02/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bayes_main_code(save_tag,NN_burn,NN_post,thin_period)
mu=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define iteration parameters based on input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NN_burn_thin=NN_burn/thin_period;    
NN_post_thin=NN_post/thin_period;    
NN=NN_burn+NN_post;                 
NN_thin=NN_burn_thin+NN_post_thin;   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare and load PSMSL annual tide gauge data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hardcoded values immediately below are those used in the paper
la1=35;     % Southern latitudinal bounds of study region
la2=46.24;  % Northern latitudinal bounds "
lo1=-80;    % Western longitudinal bounds "
lo2=-60;    % Eastern longitudinal bounds "
minnum=25;  % Minimum number of data points to consider a tide gaug record
coastcode = [960 970]; % PSMSL ID for North American northeast coast 
[DATA,LON,LAT,NAME]=prepare_data(la1,la2,lo1,lo2,minnum,coastcode);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define space and time parameters related to data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,K]=size(DATA);
D=EarthDistances([LON' LAT']); 
T=1:K; T=T-mean(T);
T0=T(1)-1;
M=sum(sum(~isnan(DATA'))~=0); % number of locations with at least one datum

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the seeds of the random number generators
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
randn('state', sum((1000+600)*clock))
rand('state', sum((1000+800)*clock))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Allocate space for the sample arrays
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
initialize_output

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the hyperparameter values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set_hyperparameters
HP = set_hyperparameters(N,K,DATA);

%%%%%%%%%%%%%%%%%%%%
% Set initial values
%%%%%%%%%%%%%%%%%%%%
set_initial_values

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up selection matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%
H_master=double(~isnan(DATA));
M_k=sum(H_master);
for k=1:K
    gauges_with_data(k).indices=find(H_master(:,k)~=0);
    selection_matrix(k).H=zeros(M_k(k),N);
    selection_matrix(k).F=zeros(M_k(k),M);
    for m_k=1:M_k(k)
       selection_matrix(k).H(m_k,gauges_with_data(k).indices(m_k))=1;
       selection_matrix(k).F(m_k,gauges_with_data(k).indices(m_k))=1;
    end
    Z(k).z=squeeze(DATA(gauges_with_data(k).indices,k));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up identity matrices and vectors of zeros or ones
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_N=eye(N);
I_M=eye(M);
ONE_N=ones(N,1);
ONE_M=ones(M,1);
ZERO_N=zeros(N,1);
ZERO_M=zeros(M,1);
for k=1:K
   I_MK(k).I=eye(M_k(k));
   ONE_MK(k).ONE=ones(M_k(k),1);
   ZERO_MK(k).ZERO=zeros(M_k(k),1);
end

tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop through the Gibbs sampler with Metropolis step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for nn=1:NN, if mod(nn,50)==0, toc, disp([num2str(nn),' of ',num2str(NN),' iterations done.']), tic, end
    nn_thin=[]; nn_thin=ceil(nn/thin_period);
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define matrices to save time
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    BMat=pi_2*eye(size(D)); invBMat=inv(BMat);
    Sig=sigma_2*exp(-phi*D); invSig=inv(Sig);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(y_K|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V_Y_K=[]; PSI_Y_K=[];
    V_Y_K=delta_2^(-1)*(selection_matrix(K).H'*(Z(K).z-selection_matrix(K).F*(l)))+...
    	invSig*(r*y(:,K-1)+(T(K)-r*T(K-1))*b);
    PSI_Y_K=(1/delta_2*selection_matrix(K).H'*selection_matrix(K).H+invSig)^(-1);
    y(:,K)=mvnrnd(PSI_Y_K*V_Y_K,PSI_Y_K)';
    clear V_Y_K PSI_Y_K   

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(y_k|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for kk=(K-1):-1:1
    	V_Y_k=[]; PSI_Y_k=[];
      	if kk==1
        	V_Y_k=1/delta_2*(selection_matrix(1).H'*(Z(1).z-selection_matrix(1).F*l))+...
                invSig*(r*(y_0+y(:,2))+(1+r^2)*T(1)*b-r*(T0+T(2))*b);
        else
         	V_Y_k=1/delta_2*(selection_matrix(kk).H'*(Z(kk).z-selection_matrix(kk).F*(l)))+...
            	invSig*(r*(y(:,kk-1)+y(:,kk+1))+(1+r^2)*T(kk)*b-r*(T(kk-1)+T(kk+1))*b);
        end
       	PSI_Y_k=inv(1/delta_2*selection_matrix(kk).H'*selection_matrix(kk).H+(1+r^2)*invSig);
       	y(:,kk)=mvnrnd(PSI_Y_k*V_Y_k,PSI_Y_k)';
      	clear V_Y_k PSI_Y_k 
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(y_0|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V_Y_0=[]; PSI_Y_0=[];
    V_Y_0=(HP.eta_tilde_y_0/HP.delta_tilde_y_0_2)*ONE_N+invSig*(r*y(:,1)-r*(T(1)-r*T0)*b);
    PSI_Y_0=inv(1/HP.delta_tilde_y_0_2*I_N+r^2*invSig);
    y_0=mvnrnd(PSI_Y_0*V_Y_0,PSI_Y_0)';
    clear V_Y_0 PSI_Y_0
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(b|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   	V_B=[]; PSI_B=[]; SUM_K=ZERO_N;
    for kk=1:K
     	if kk==1
         	SUM_K=SUM_K+(T(1)-r*T0)*(y(:,1)-r*y_0);
        else
          	SUM_K=SUM_K+(T(kk)-r*T(kk-1))*(y(:,kk)-r*y(:,kk-1));
        end
   	end
    V_B=mu*invBMat*ONE_N+invSig*SUM_K;
    PSI_B=inv(invBMat+invSig*sum((T-r*[T0 T(1:K-1)]).^2));
    b=mvnrnd(PSI_B*V_B,PSI_B)';
    clear V_B PSI_B SUM_K

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(mu|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V_MU=[]; PSI_MU=[];
    V_MU=HP.eta_tilde_mu/HP.delta_tilde_mu_2+ONE_N'*invBMat*b;
   	PSI_MU=inv(1/HP.delta_tilde_mu_2+ONE_N'*invBMat*ONE_N);
    mu=normrnd(PSI_MU*V_MU,sqrt(PSI_MU));
    clear V_MU PSI_MU  
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(pi_2|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    inside1=[]; inside2=[];
    inside1=N/2;
    inside2=1/2*((b-mu*ONE_N)'/(eye(size(D))))*(b-mu*ONE_N);
    pi_2=1/randraw('gamma', [0,1/(HP.nu_tilde_pi_2+inside2),...
      	(HP.lambda_tilde_pi_2+inside1)], [1,1]);
   	clear inside*
    
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(delta_2|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SUM_K=0;
    for kk=1:K
      	xxx=[]; xxx=(Z(kk).z-selection_matrix(kk).H*y(:,kk)-selection_matrix(kk).F*(l));
      	SUM_K=SUM_K+xxx'*xxx;
   	end
    delta_2=1/randraw('gamma', [0,1/(HP.nu_tilde_delta_2+1/2*SUM_K),...
     	(HP.lambda_tilde_delta_2+1/2*sum(M_k))], [1,1]);    
    clear SUM_K
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(r|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V_R=0; PSI_R=0;
    for kk=1:K
     	if kk==1
         	V_R=V_R+((y_0-b*T0)')*invSig*(y(:,1)-b*T(1));
         	PSI_R=PSI_R+((y_0-b*T0)')*invSig*(y_0-b*T0);
        else
         	V_R=V_R+((y(:,kk-1)-b*T(kk-1))')*invSig*(y(:,kk)-b*T(kk));
          	PSI_R=PSI_R+((y(:,kk-1)-b*T(kk-1))')*invSig*(y(:,kk-1)-b*T(kk-1));
        end        
   	end
    PSI_R=inv(PSI_R);
    dummy=1;
    while dummy
      	sample=normrnd(PSI_R*V_R,sqrt(PSI_R));
      	if sample>HP.u_tilde_r&&sample<HP.v_tilde_r
         	r=sample;
          	dummy=0;
        end
    end
    clear V_R PSI_R dummy ctr

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(sigma_2|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    RE=[]; invRE=[]; SUM_K=0;
    RE=exp(-phi*D);
   	invRE=I_N/RE;
    for kk=1:K
     	if kk==1
         	DYKK=[];
          	DYKK=y(:,1)-r*y_0-(T(1)-r*T0)*b;
         	SUM_K=SUM_K+(DYKK')*invRE*DYKK;           
        else
         	DYKK=[];
           	DYKK=y(:,kk)-r*y(:,kk-1)-(T(kk)-r*T(kk-1))*b;
         	SUM_K=SUM_K+(DYKK')*invRE*DYKK;           
        end
    end
   	sigma_2=1/randraw('gamma', [0,1/(HP.nu_tilde_sigma_2+1/2*SUM_K),...
     	(HP.lambda_tilde_sigma_2+N*K/2)], [1,1]);
   	clear RE invRE SUM_K DYKK
            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(phi|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Phi_now=log(phi);
    Phi_std=0.05;
    Phi_prp=normrnd(Phi_now,Phi_std);
    R_now=exp(-exp(Phi_now)*D);
    R_prp=exp(-exp(Phi_prp)*D);
    invR_now=inv(R_now);
    invR_prp=inv(R_prp);
    sumk_now=0;
    sumk_prp=0;
        
   	for kk=1:K
      	if kk==1
         	DYYK=y(:,1)-r*y_0-(T(1)-r*T0)*b;
          	sumk_now=sumk_now+(DYYK')*invR_now*DYYK;
          	sumk_prp=sumk_prp+(DYYK')*invR_prp*DYYK;
        else
         	DYYK=y(:,kk)-r*y(:,kk-1)-(T(kk)-r*T(kk-1))*b;
         	sumk_now=sumk_now+(DYYK')*invR_now*DYYK;
         	sumk_prp=sumk_prp+(DYYK')*invR_prp*DYYK;
        end
    end
        
 	ins_now=-1/(2*HP.delta_tilde_phi_2)*(Phi_now-HP.eta_tilde_phi)^2-1/(2*sigma_2)*sumk_now;
   	ins_prp=-1/(2*HP.delta_tilde_phi_2)*(Phi_prp-HP.eta_tilde_phi)^2-1/(2*sigma_2)*sumk_prp;
  	MetFrac=det(R_prp*invR_now)^(-K/2)*exp(ins_prp-ins_now);
   	success_rate=min(1,MetFrac);
   	if rand(1)<=success_rate
     	Phi_now=Phi_prp; 
    end
  	phi=exp(Phi_now);
  	clear phi_now phi_std phi_prp mat_now mat_prp inside sumk MetFrac success_rate 
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(l|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V_L=[]; PSI_L=[]; SUM_K1=ZERO_M; SUM_K2=zeros(M,M);
    for kk=1:K
     	SUM_K1=SUM_K1+(selection_matrix(kk).F')*(Z(kk).z-selection_matrix(kk).H*y(:,kk));
        SUM_K2=SUM_K2+(selection_matrix(kk).F')*selection_matrix(kk).F;
    end
    V_L=nu/tau_2*ONE_M+1/delta_2*SUM_K1;
    PSI_L=inv(1/tau_2*I_M+1/delta_2*SUM_K2);
    l=mvnrnd(PSI_L*V_L,PSI_L)';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(nu|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V_NU=[]; PSI_NU=[];
   	V_NU=HP.eta_tilde_nu/HP.delta_tilde_nu_2+1/tau_2*(ONE_M'*l);
    PSI_NU=inv(1/HP.delta_tilde_nu_2+M/tau_2);
    nu=normrnd(PSI_NU*V_NU,sqrt(PSI_NU));
    clear V_NU PSI_NU
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample from p(tau_2|.)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tau_2=1/randraw('gamma', [0,1/(HP.nu_tilde_tau_2+...
     	1/2*(((l-nu*ONE_M)')*(l-nu*ONE_M))),(HP.lambda_tilde_tau_2+M/2)], [1,1]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now update arrays
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    update_all_arrays

end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% delete the burn-in period values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delete_burn_in

%%%%%%%%%%%%%
% save output
%%%%%%%%%%%%%
if exist('bayes_model_solutions')==0
 mkdir('bayes_model_solutions')
end

save(['bayes_model_solutions/experiment_',save_tag,'.mat'],...
    'MU','NU','PI_2','DELTA_2','SIGMA_2','TAU_2',...
    'PHI','B','L','R','Y_0','Y','HP','DATA','LON','LAT',...
    'NAME','N','K','D')