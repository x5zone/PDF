clc;
close all;
clear all;

N = 1000;

r1 = 3;
r2 = 6;
r3 = 9;

inexA = [];

outexA = [];

for i = 1:1.5*N
    
    inex = (2*rand(2,1)-1)*r3;
    
    d = norm(inex);
    
    if  (d <= r3)
        
        inexA = [inexA inex];
        
        if ( (d< r1) && inex(2)>0 ) || ( (d< r2) && (d>r1) &&  inex(2)<0 ) || ( (d > r2) && inex(2)>0 ) 
                      
            outex = 1;
        else
            outex = -1;
        end;
        
        outexA = [outexA outex];
        
    end;
    
end;

seq = [inexA;outexA];

seq(:,N+1:end) = [];

save ('C:\Haran\Research\NN\Code_NN\Ch15_Haykin\Data\seq', 'seq');

%==========================================================================
%plot results
%==========================================================================

for i=1:N/2
    if seq(3,i) == 1
        plot(seq(1,i),seq(2,i),'.r','markersize',5);
    else
        plot(seq(1,i),seq(2,i),'.b','markersize',5);
    end;
    hold on;
end;

t = [pi : 2*pi/200 : 2*pi];
x = r1*[cos(t); sin(t)];
plot(x(1,:), x(2,:),'k','linewidth',2);

t = [0 : 2*pi/200 : 2*pi];
x = r2*[cos(t); sin(t)];
plot(x(1,:), x(2,:),'k','linewidth',2);

x = r3*[cos(t); sin(t)];
plot(x(1,:), x(2,:),'k','linewidth',2);

line([-r3,-r1],[0,0],'color','black'); line([r1,r3],[0,0],'color','black');
axis([-9.1 9.1 -9.1 9.1]);

hold off;
