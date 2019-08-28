%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% autocorrelation
% Sample autocorrelation of a time series Y(t) at lag k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code written by CGP 2017/02/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rk=autocorrelation(Y,k)

n=numel(Y);
kp1=k+1;
Ybar=mean(Y);

top=0; bot=0;

for t=kp1:n
    top=top+(Y(t)-Ybar)*(Y(t-k)-Ybar);
end
for t=1:n
    bot=bot+(Y(t)-Ybar)^2;
end

rk=top/bot;

return