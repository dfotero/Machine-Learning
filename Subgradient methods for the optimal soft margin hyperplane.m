clear all; 
close all;
load nuclear

rng(0);

lambda=0.001;

iteration=1;

error=1;

newX=[ones(1,20000); x];
parameters= ((newX*newX')^(-1))*newX*y';

b=parameters(1,1);
w=parameters(2:3,1);
J=[];
lastJ=0;
theta=[b; w];

while error >0.001
    
    alpha=100/iteration;
    
    gW=zeros(2,1);
    gB=0;
    theJ=0;
    
    for i=1:20000
    
        if y(1,i)*(w'*x(:,i)+b)<1
            
            gW=gW-y(1,i)*x(:,i);
            gB=gB-y(1,i);
            theJ=theJ+1-y(1,i)*(w'*x(:,i)+b);
        end 
        
    end
    
    gW=gW+lambda*w;
    
    newJ=theJ/20000+lambda/2*norm(w)^2;
    J=[J newJ];
    newTheta=theta-[alpha*gB;alpha*gW];
    
    b=newTheta(1,1);
    w=newTheta(2:3,1);
    
    error=sum(abs(theta-newTheta))/20000;
    
    theta=newTheta;
    lastJ=newJ;
    iteration=iteration+1;
    
end

theta
plot(J);

scatter3(x(1,:),x(2,:),y)
hold on
[x1, x2]=meshgrid(0:0.1:8,0:0.05:1.5);
ytest=w(1,1)*x1+w(2,1)*x2+b;
surf(x1,x2,ytest)

