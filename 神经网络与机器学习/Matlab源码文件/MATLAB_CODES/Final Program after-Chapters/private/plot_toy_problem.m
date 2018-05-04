function plot_toy_problem(X,Y,Xt,Yt,oo)
% PLOT_TOY_PROBLEM(Xl,Yl,Xu,Xt,Yt)
% plot a 2D toy problem. 
% Xl,Yl: labeled examples, Xu unlabeled, Xt test, Yt: predicted labels
% Yt, if not empty  is the predicted (real) output of the test points.

Xl=X(Y~=0,:);
Yl=Y(Y~=0);
Xu=X(Y==0,:);


 xmin=min(Xt(:,1)); 
 xmax=max(Xt(:,1)); 
 ymin=min(Xt(:,2)); 
 ymax=max(Xt(:,2));
 
 cla;
 set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);
 hold on
 
 if ~isempty(Yt)
   [x,y] = meshgrid(xmin:(xmax-xmin)/50:xmax,ymin:(ymax-ymin)/50:ymax);
   Xt2=[reshape(x,prod(size(x)),1) reshape(y,prod(size(y)),1)];
   if ~isequal(Xt,Xt2)
     error('Incorrect Xt');
   end;
   
   Yt=reshape(Yt,size(x));
   
   sp = pcolor(x,y,Yt);
   load red_black_colmap;
   colormap(red_black);
   shading flat;
   %shading interp
   set(sp,'LineStyle','none');
   axis off
   %colormap((1+gray)/2);
   %colormap(cool)
   [dummy c1] = contour(x,y,Yt,[0 0],'k');
   set(c1,'LineWidth',1);
   %[dummy c2] = contour(x,y,Yt,[-1 1],'k:');
 end;

 for i=1:size(Xu,1)
   if oo(i) == 1,  
   plot(Xu(i,1),Xu(i,2),'rx','Markersize',7);
   else
   plot(Xu(i,1),Xu(i,2),'k+','Markersize',7);
   end
 end;
 
 for i=1:length(Yl)
   if Yl(i)==1
     plot(Xl(i,1),Xl(i,2),'^w','Linewidth',2,'MarkerSize',10);
   else
     plot(Xl(i,1),Xl(i,2),'ow','Linewidth',2,'MarkerSize',10);
   end;
 end;
 
 axis on;
 box on;
 
 hold off;