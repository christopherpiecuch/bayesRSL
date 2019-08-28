%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [DATA,X,Y,NAME] = prepare_data(la1,la2,lo1,lo2,minnum,coastcode)
% Loads annual sea level data from specified geographic region and
% coastline from the PSMSL database into into a [NxK] matrix, where N is
% the number of tide gauges and K is the number of years of the data
% record; lack of data are filled with NaNs.
% INPUT: 
%   la1         Southern latitudinal bounds of study region
%   la2         Northern latitudinal bounds "
%   lo1         Western longitudinal bounds "
%   lo2         Eastern longitudinal bounds "
%   minnum      Minimum number of data points to consider a gauge record
%   coastcode   PSMSL coastline ID  
% OUTPUT:
%   DATA        [NxK] array of sea level values in units of m
%   X           Longitude of N tide gauges
%   Y           Latitude of N tide gauges
%   NAME        Name of N tide gauge sites
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code written by CGP 2017/02/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DATA,X,Y,NAME] = prepare_data(la1,la2,lo1,lo2,minnum,coastcode)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load annual tide gauge data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dircur=pwd;
cd rlr_annual % switch to PSMSL annual data directory
 % The above directory was downloaded from www.psmsl.org/data/ 
 % See READ_ME.rtf and PSMSL website for more recent data
 data = readAnnual([dircur,'/rlr_annual/']);
cd ..

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Delete empty entries in database or records not along specified
% coastlines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=numel(data):-1:1
    if (strcmp('',data(n).name)||~ismember(data(n).coastline,coastcode))
        data(n)=[]; 
    end     
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Delete records outside the specified geographic region or with fewer than
% "minnum" data points.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=numel(data):-1:1
   if sum(~isnan(data(n).height))<=minnum 
       data(n)=[];
   elseif (data(n).latitude<la1)||(data(n).latitude>la2)||(data(n).longitude<lo1)||(data(n).longitude>lo2)
      data(n)=[]; 
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sort remaining records by latitude
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:numel(data)
 lat2sort(n)=data(n).latitude;
end
[Y,I]=sort(lat2sort);
data2=data(I);
clear data
data=data2; clear data2 I Y lat2sort

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Be careful of flagged or interpolated values in PSMSL database. Make them
% into NaNs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:numel(data)
    iTakeOut=[]; iTakeOut=find((data(n).dataflag)|(data(n).interpolated));
    data(n).height(iTakeOut)=nan;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert data structure into matrix array structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=numel(data);
for n=1:N
    if n==1
        T_0=data(n).year(1);
        T_F=data(n).year(numel(data(n).year));
    else
        if T_0>data(n).year(1)
           T_0=data(n).year(1);
        end
        if T_F<data(n).year(numel(data(n).year))
           T_F=data(n).year(numel(data(n).year));
        end
    end
end
K=T_F-T_0+1;
T=T_0:T_F;
DATA=nan(N,K);
for n=1:N
   t_0=find(T==data(n).year(1));
   t_f=find(T==data(n).year(numel(data(n).year)));
   DATA(n,t_0:t_f)=1e-3*data(n).height; % Convert data from mm to m
   X(n)=data(n).longitude;
   Y(n)=data(n).latitude;
   NAME(n).name=data(n).name;
   clear t_0 t_f
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONAL: delete values prior to the beginning of the continuous record
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=find(sum(~isnan(DATA))==0,1,'last')+1; 
DATA(:,1:(i-1))=[];
T(1:(i-1))=[];
T_0=T(1);
N=numel(data);

return