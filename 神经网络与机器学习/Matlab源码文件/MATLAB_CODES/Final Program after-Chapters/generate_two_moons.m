function [X,Y,Xt]=generate_two_moons(r,w,d,N,seed,display)
% GENERATE_TWO_MOONS 
% INPUTS:
% r: radius of 2 moons
% w: width of the 2 moons
% d: distance between the two moons
% N: number of samples to generate in each moon
% display: 1 shows the 2moons after generation, 0 no display
%
% OUTPUTS:
% X: 2N X 2 matrix 
% Y: 2N X 1 vector -- +1 for top moon -1 for bottom moon
% Xt: uniformly spaced points on a grid
%
% Vikas Sindhwani (vikass@cs.uchicago.edu)


% top moon
rand('seed',seed);
r = r+w/2;

% randomly pick radial coordinate between r-w and r 
R=(r-w)*ones(N,1) + rand(N,1)*w;
% randomly pick the angular coordinate
theta=rand(N,1)*pi;

X=[R.*cos(theta) R.*sin(theta)];
Y=ones(N,1);

% bottom moon

% randomly pick radial coordinate between r-w and r 
R=(r-w)*ones(N,1) + rand(N,1)*w;
% randomly pick the angular coordinate
theta=pi+rand(N,1)*pi;

% move x coordinate by y coordinate down by d
del_x=r-(w/2);
del_y=-d;
x=[R.*cos(theta)+del_x R.*sin(theta)+del_y]
y=-ones(N,1);

X=[X;x];
Y=[Y;y];

r=r-w/2;

if display==1
   plot(X(Y==1,1),X(Y==1,2),'r+','MarkerSize',8); hold on;
   plot(X(Y==-1,1),X(Y==-1,2),'bx','MarkerSize',8); 
   axis tight;
   axis equal;
   title(['2 moons dataset : r=' num2str(r) ...
       ' w=' num2str(w) ' d=' num2str(d) ' N=' num2str(N)]);
end

 xmin=min(X(:,1)); 
 xmax=max(X(:,1)); 
 ymin=min(X(:,2)); 
 ymax=max(X(:,2));
 
[x,y] = meshgrid(xmin:(xmax-xmin)/50:xmax,ymin:(ymax-ymin)/50:ymax);
Xt=[reshape(x,prod(size(x)),1) reshape(y,prod(size(y)),1)];