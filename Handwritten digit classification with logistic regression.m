%---------------------------------
% Handwritten digit classification with logistic regression
% Author: Daniel Felipe Otero Leon
%---------------------------------


%----------------------------
% Import databse
%----------------------------

load mnist_49_3000
[d,n] = size(x);

%----------------------------
% Initialize parameters
%----------------------------

xTraining = x(:,1:2000); 
yTraining = y(:,1:2000);

newX = [ones(2000,1) xTraining'];

theta=zeros(785,1);

lambda=10;

%-----------------------------
% Newton's Method
%-----------------------------

stopCriteria=0.0000001;
theDifference=1000;
objFunction=0;

while theDifference >= stopCriteria
    
    theHessian=hessianCalc(theta,newX,y,lambda);
    
    newTheta=theta-(theHessian^(-1))*gradientCalc(theta,newX,y,lambda);     
    
    theDifference=norm(theta-newTheta);
    
    theta=newTheta;

end

%-----------------------------
%Test
%-----------------------------

xTest = [ones(1000,1) x(:,2001:3000)']; 
yTest = y(:,2001:3000);

prediction=zeros(1000,3);


for i= 1:1000
    
    nabla=1/(1+exp((theta')*(xTest(i,:)')));

    prediction(i,2)=nabla;
    prediction(i,3)=i;
    if nabla >= 0.5
        
        prediction(i,1)=-1;
    
    else
        
        prediction(i,1)=1;

    end
end 

%----------------------------------
%Error Estimation
%----------------------------------

theError=prediction(:,1)-yTest';
error= 1-sum(theError==0)/1000;
error

objFunction=lambda*norm(theta);

for i=2000
    
objFunction=objFunction+log(1+exp(-yTraining(i)*(theta')*(newX(i,:)')));

end

objFunction

%----------------------------
% Print Numbers
%----------------------------

theErrors=prediction-[yTest' 0.5*ones(1000,1) zeros(1000,1)];
theErrors=abs(theErrors);

sortedTheErrors = sortrows(theErrors,[1,2,3],'descend');

imageArray=[];

figure;
for i=1:20

    subplot(4,5,i);
    imagesc(reshape(x(:,2000+sortedTheErrors(i,3)),[sqrt(d),sqrt(d)])')
    
    if y(2000+sortedTheErrors(i,3))==1
        
        title('9')
        
    else
        
         title('4')
        
    end
    
end

% notice the transpose

%----------------------------
% function: Estimate the gradient for a particular theta
% parameters: 
% - theta: theta in t
% - newX: training matrix x with vector of 1
% - y: training vector y
% - lambda: regularization value
% returns: gradient for theta in t
%----------------------------

function gradient=gradientCalc(theta,newX,y,lambda)

    gradient=2*lambda*theta;

    for i=1:2000
   
        gradient=gradient-y(i)*exp(-y(i)*(theta')*(newX(i,:)'))*(newX(i,:)')/(1+exp(-y(i)*(theta')*(newX(i,:)')));
    
    end

end

%----------------------------
% function: Estimate the Hessian for a particular theta
% parameters: 
% - theta: theta in t
% - newX: training matrix x with vector of 1
% - y: training vector y
% - lambda: regularization value
% returns: hessian for theta in t
%----------------------------

function hessian=hessianCalc(theta,newX,y,lambda)

    hessian=2*lambda*eye(785);

    for i=1:2000
   
        hessian=hessian+(y(i)^2)*(newX(i,:)')*newX(i,:)*exp(-y(i)*(theta')*(newX(i,:)'))/(1+exp(-y(i)*(theta')*(newX(i,:)')))^2;
    
    end

end
