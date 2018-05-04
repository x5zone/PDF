function plotDecisionBdy(seq)

global r1 r2 r3;

X = seq(1,:);

Y = seq(2,:);

class = seq(3,:);

%==========================================================================
% shade estimated decision region
%==========================================================================

colors = ['kx'; 'r+'];

%figure;

for i = 1:2
    if i == 1
        j = 0;
    else
        j = 1;
    end;
    thisX = X(class == j );
    thisY = Y(class == j );
    h = plot(thisX, thisY,colors(i,:),'markerfacecolor',colors(i,1), 'markeredgecolor',colors(i,1));
    set(h,'markersize',5);
    hold on;
end;

xlabel('x'); ylabel('y');


axis([-9 9 -9 9]);

axis square;


% %==========================================================================
% % draw desired bdy
% %==========================================================================
% 
% t = [-pi: 2*pi/200 :0];
% x = r1*[cos(t); sin(t)];
% plot(x(1,:), x(2,:),'k','linewidth',2.5);
% % hold on;
% 
% t = [0 : 2*pi/200 : 2*pi];
% x = r2*[cos(t); sin(t)];
% plot(x(1,:), x(2,:),'k','linewidth',2.5);
% 
% x = r3*[cos(t); sin(t)];
% plot(x(1,:), x(2,:),'k','linewidth',2.5);

% line([-r3,-r1], [0,0],'color','black','linewidth',2.5); 
% 
% line([r1,r3],[0,0],'color','black','linewidth',2.5);
% 
% 
% %draw sqaure
% line([-9,9], [-9,-9],'color','black'); 
% line([9,9],[-9,9],'color','black');
% line([-9,9], [9,9],'color','black'); 
% line([-9,-9],[-9,9],'color','black');

axis([-9.1 9.1 -9.1 9.1]);