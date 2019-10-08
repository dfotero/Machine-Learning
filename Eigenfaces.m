load yalefaces % loads the 3-d array yalefaces
data=[];
for i=1:size(yalefaces,3)
    x = double(yalefaces(:,:,i));
    %imagesc(x);
    %colormap(gray)
    %drawnow% pause(.1)
    data=[data,reshape(x,2016,1)];
end

theMean=mean(data,2);

S=0;

for i=1:size(yalefaces,3)
   
    theX=data(:,i)-theMean;
    S=S+theX*theX';
    
end

S=S/2414;


theE=[1:1:2016];
theE=theE';
theE=[theE eig(S)];
theE=sortrows(theE,2,'descend');

figure
semilogy(theE(:,2))

theVar95=0;
k95=0;

while theVar95<0.95
   
    k95=k95+1;
    theVar95=theVar95+theE(k95,2)/sum(theE(:,2)); 
    
end

theVar99=theVar95;
k99=k95;

while theVar99<0.99
   
    k99=k99+1;
    theVar99=theVar99+theE(k99,2)/sum(theE(:,2)); 
    
end

k95
k99

perRiduction95=1-k95/2016
perRiduction99=1-k99/2016

[theVectors,D]=eig(S);

subplot(4,5,1);
a=reshape(theMean,48,42);
imagesc(a);
colormap(gray)

for i=1:19
   subplot(4,5,i+1);
   theV=theVectors(:,theE(i,1));
   theV=reshape(theV,48,42);
   imagesc(theV);
   colormap(gray)
end
