
load bodyfat_data

xTraining = X(1:150,:);
yTraining=y(1:150,:);

xTest=X(151:248,:);
yTest=y(151:248,:);

lambda=10;

newX= [ones(150,1) xTraining];

parameters= ((newX'*newX + 150*lambda*eye(3))^(-1))*newX'*yTraining;

newX= [ones(98,1) xTest];

forecastY = newX*parameters;

errors=forecastY-yTest;

mse=sum(errors.^2)/98;

parameters
mse
x=[1 100 100];
x*parameters


