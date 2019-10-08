%-------------------------------
% KRR without offset
%-------------------------------

load bodyfat_data

xTraining = X(1:150,:);
yTraining=y(1:150,:);

xTest=X(151:248,:);
yTest=y(151:248,:);

sig=15;
lambda=0.003;

newX=  xTraining;

Onn(1:150,1:150)=1/150;
K=zeros(150,150);

for i=1:150
   
    for j=1:150
       
        K(i,j)= exp(-dist2(newX(i,:),newX(j,:))/(2*(sig^2)));
        
    end
    
end

Onn(1:150,1:150)=1/150;

Kh=K-K*Onn-Onn*K+Onn*K*Onn;

yh=yTraining-mean(y);


KO=K*Onn;
OKO=Onn*K*Onn;
kx=KO(:,1)/150-OKO(:,1)/(150^2);

wtx=(yh')*((Kh+150*lambda*eye(150))^(-1))*kx;

yTrainPred=(yh')*((Kh+150*lambda*eye(150))^(-1))*Kh;

errorTraining=sum((yTraining-yTrainPred').^2)/150


Kt=zeros(150,98);

newXt= xTest;

for i=1:150
   
    for j=1:98
       
        Kt(i,j)= exp(-dist2(newX(i,:),newXt(j,:))/(2*(sig^2)));
        
    end
    
end

Onm(1:150,1:98)=1/150;

Kht=Kt-K*Onm-Onn*Kt+Onn*K*Onm;

ypred=(yh')*((Kh+150*lambda*eye(150))^(-1))*Kht

errorTest=sum((yTest-ypred').^2)/98

function n2 = dist2(x, c)
%DIST2     Calculates squared distance between two sets of points.
%
%     Description
%     D = DIST2(X, C) takes two matrices of vectors and calculates the
%     squared Euclidean distance between them.  Both matrices must be of
%     the same column dimension.  If X has M rows and N columns, and C has
%     L rows and N columns, then the result has M rows and L columns.  The
%     I, Jth entry is the  squared distance from the Ith row of X to the
%     Jth row of C.
%
%     See also
%     GMMACTIV, KMEANS, RBFFWD
%
%     Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)
[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);

if dimx ~= dimc
    error('dist2.m: Data dimension does not match dimension of centres')
end
n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
        ones(ndata, 1) * sum((c.^2)',1) - ...
        2.*(x*(c'));
end
