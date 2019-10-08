%---------------------------------
% Problem 2 HW 2
% Author: Daniel Felipe Otero Leon
%---------------------------------


%----------------------------
% Import databse
%----------------------------

z = dlmread('spambase.data',',');
rng(0); % initialize the random number generator
rp = randperm(size(z,1)); % random permutation of indices
z = z(rp,:); % shuffle the rows of a
x = z(:,1:end-1);
y = z(:,end);

%--------------------------------
% Divide data to training and test
%--------------------------------

xTraining = x(1:2000,:);
yTraining=y(1:2000,:);

xTest=x(2001:4601,:);
yTest=y(2001:4601,:);

%--------------------------------
%Estimate the marginal probability for Y
%--------------------------------

prIsSpam=sum(yTraining)/2000;
prNotSpam=(2000-sum(yTraining))/2000;

%----------------------------------
%Quantize the x
%----------------------------------

theMedians= median(xTraining);
quantX=zeros(2000,57);

for i=1:2000
    for j=1:57
        if xTraining(i,j) <=theMedians(j)
           quantX(i,j)=1;
        else
           quantX(i,j)=2;
        end
    end
end

%----------------------------------
%Estimate the marginal probability for X
%----------------------------------
 
prX=zeros(57,2);

prX(:,1)=transpose(sum(quantX==1))/2000;
prX(:,2)=transpose(sum(quantX==2))/2000;

%----------------------------------
%Estimate conditional probability for X
%----------------------------------

prXY=zeros(57,2,2); %where prXY(i,j,k) represents the P(Xi=j|Y=k-1)

for i=1:57
    prXY(i,1,1)=sum(quantX(yTraining==0,i)==1)/length(quantX(yTraining==0,i));
    prXY(i,2,1)=sum(quantX(yTraining==0,i)==2)/length(quantX(yTraining==0,i));
    
    prXY(i,1,2)=sum(quantX(yTraining==1,i)==1)/length(quantX(yTraining==1,i));
    prXY(i,2,2)=sum(quantX(yTraining==1,i)==2)/length(quantX(yTraining==1,i)); 
end

%----------------------------------
%Clasify the test data
%----------------------------------

quantXTest=zeros(2601,57);

for i=1:2601
    for j=1:57
        if xTest(i,j) <=theMedians(j)
            quantXTest(i,j)=1;
        else            
            quantXTest(i,j)=2;            
        end        
    end    
end

yForecastPr=zeros(2601,2);
yForecast=zeros(2601,1);

for i=1:2601   
    NotSpam = prNotSpam;
    IsSpam = prIsSpam;    
    
    for j=1:57        
        NotSpam=NotSpam*prXY(j,quantXTest(i,j),1);
        IsSpam=IsSpam*prXY(j,quantXTest(i,j),2);        
    end
    
    yForecastPr(i,1)=NotSpam;
    yForecastPr(i,2)=IsSpam;
    
    if yForecastPr(i,1) > yForecastPr(i,2)    
        yForecast(i)=0;        
    else        
        yForecast(i)=1;        
    end
end


%----------------------------------
%Error Estimation
%----------------------------------

theError=yForecast-yTest;
error=1-sum(theError==0)/2601;
error