function Dists_GC_Mat=EarthDistances(llPoints)
% function Dists=EarthDistances(llPoints)
%Input is N by 2, each row a (long, lat) pair, -180<long<180; -90<lat<90.
%Output is a N by N matrix of great circle distances in KM (approximating the
%earth as a sphere), where the (i,j) entry is the distance between the ith
%and jth rows of the input vector. So the diagonal is zero. 
%This makes use of the so-called haversine formulation (see wikipedia),
%which is also used in the m_lldist.m code of the m_map package. (m_lldist
%gives identical results, but doesn't seem well set up for the formulation
%of the matrix we want here.)
%radius of the earth si taken as 6378.137
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CGP 2017/02/21. This code originally written by Martin P. Tingley and
% included in supporting material to Tingley, M. P., and P. Huybers (2010),
% A Bayesian Algorithm for Reconstructing Climate Anomalies in Space and 
% Time. Part I: Development and Applications to Paleoclimate Reconstruction 
% Problems, J. Climate, 23, 2759-2781.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


RR=6378.137; %radius of the earth in km
NN=length(llPoints(:,1));
%make a NN^2 by 4 matrix. Each row one of the NN^2 possible sets of
%two points:

Pts_paired_Vec=[kron(llPoints, ones(NN,1)), kron(ones(NN,1), llPoints)];
Dists_GC_AsVec=RR*2*asin(sqrt(sin((Pts_paired_Vec(:,2)-Pts_paired_Vec(:,4))*pi/180/2).^2 + cos(Pts_paired_Vec(:,2)*pi/180).*cos(Pts_paired_Vec(:,4)*pi/180).*sin(abs(Pts_paired_Vec(:,1)-Pts_paired_Vec(:,3))*pi/180/2).^2));
Dists_GC_Mat=reshape(Dists_GC_AsVec, NN, NN);

