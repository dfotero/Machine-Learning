%---------------------------------
% Problem 2 HW 3
% Author: Daniel Felipe Otero Leon
%---------------------------------

clear all; close all;

n = 200;

rng(0); % seed random number generator

x = rand(n,1);

z = zeros(n,1); 
k = n*0.4; 
rp = randperm(n); 
outlier_subset = rp(1:k); 
z(outlier_subset)=1; % outliers

y = (1-z).*(10*x + 5 + randn(n,1)) + z.*(20 - 20*x + 10*randn(n,1));
% plot data and true line
scatter(x,y,'b')
hold on
t = 0:0.01:1;
plot(t,10*t+5,'k')
% add your code for ordinary least squares below

newX= [ones(n,1) x];
paramOLS= ((newX'*newX)^(-1))*newX'*y

plot(t, paramOLS(2,1)*t + paramOLS(1,1), 'g--');

% add your code for the robust regression MM algorithm below

b=paramOLS(1,1);
w=paramOLS(2,1);


error=1;

while error > 0.000001
    
   rt=y-w*x-b;
   c= (1+rt.^2).^(1-2);
   
   [newW, newB] =wls(newX,y,c);
   
   error=abs(w-newW)+abs(b-newB);
   
   w=newW;
   b=newB;

end

paramsMM=[b w]'

plot(t, w*t + b, 'r:');
legend('data','true line','least squares','robust')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a helper function to solve weighted least squares
function [w,b] = wls(x,y,c)

    C=diag(c);
    theta= ((x'*C*x)^(-1))*x'*C*y;
    
    b=theta(1,1);
    w=theta(2,1);
    
end

