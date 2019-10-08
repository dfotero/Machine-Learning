clear all
close all

load bodyfat_data

xTraining = X(1:150,:);
yTraining=y(1:150,:);

xTest=X(151:248,:);
yTest=y(151:248,:);

rng(0);

%Parameters
n=150;
W1=normrnd(0,1,[2,64]);
W2=normrnd(0,1,[64,16]);
W3=normrnd(0,1,[16,1]);
b1=zeros(64,1);
b2=zeros(16,1);
b3=zeros(1,1);
gamma=5*10^(-9);

tError=10000;
dif=1;

%Gradient Descent 

while dif >0.0001
    
[a1,a2,theY]=forwardPass(xTraining,n, W1,W2,W3,b1,b2,b3);
theError= mse(theY',yTraining);

dif=abs(tError-theError);

tError=theError;

z1=max(0,a1);
z2=max(0,a2);

[nW1,nW2,nW3,nb1,nb2,nb3] = backwardPass(xTraining, yTraining,n, a1,a2,z1,z2,theY,W1,W2,W3,b1,b2,b3,gamma);

W1=nW1;
W2=nW2;
W3=nW3;
b1=nb1;
b2=nb2;
b3=nb3;

end

%Test Data

[a1,a2,theY]=forwardPass(xTest,98, W1,W2,W3,b1,b2,b3);

%Errors

tError %Training errors
testError=mse(theY',yTest) %Test errors


%Forward Pass

function [a1,a2,theY] = forwardPass(x,n, W1,W2,W3,b1,b2,b3)

B1=repmat(b1,1,n);
B2=repmat(b2,1,n);
B3=repmat(b3,1,n);

a1=(W1')*x'+B1;
z1=max(0,a1);
a2=(W2')*z1+B2;
z2=max(0,a2);
theY=(W3')*z2+B3;

end

%Backward Pass

function [nW1,nW2,nW3,nb1,nb2,nb3] = backwardPass(xTraining, yTraining,n, a1,a2,z1,z2,theY,W1,W2,W3,b1,b2,b3,gamma)

d3=-2*(yTraining-theY');

d2=(d3*(W3')).*max(0,sign(a2)');

d1=(d2*(W2')).*max(0,sign(a1)');

nW1=W1-(gamma/n)*(xTraining')*d1;

nW2=W2-(gamma/n)*(z1)*(d2);

nW3=W3-(gamma/n)*(z2)*(d3);

nb1=b1-(gamma/n)*(d1')*ones(150,1);

nb2=b2-(gamma/n)*(d2')*ones(150,1);

nb3=b3-(gamma/n)*(d3')*ones(150,1);
end


