function W = NN_GHA_A(image,W_other,mval_other)
% Exp 4: NN_GHA_A GHA appendix program
% Modified from Hugh Pasika's program
% W is the output weights for compress other images
% this program is used with NN_GHA
% such as
% [W_out, mval_other] = NN_GHA('lina.tif');
% W = NN_GHA_A('pepper.tif',W_out, mval_other);

% Copyright
% Yanbo Xue
% Adaptive Systems Laboratory
% McMaster University
% yxue@soma.mcmaster.ca
% April 6, 2007
% modified on March 29, 2007 -- Adding new boundary function
% modified from Hugh Pasika 1997

%%===================== Initializtion =====================================
epochs = 2000;
pcs    = 8;  % number of principal components
disp_w_row = 4; % the weights are displayed in (disp_w_row) * (disp_w_col)
disp_w_col = 2; 
num_layer  = 1; % number of image layer
mu = 0.0001; % learning rate
rm = 8; % block size 8*8 row 
cm = 8; % column
r  = 32; % r*rm = the height of image
c  = 32; % c*cm = the width of image
%p  = rm*cm; 	
%n = [6 6 5 5 4 4 4 2 2 2 2 2]; %%quantization bit rates
%n = [5 5 3 2 2 2 2 2]; %%quantization bit rates
%n = [6 6 6 6 5 5 5 5];
%n = [7 7 7 7 5 5 5 5];
%n = [6 6 3 3 3 3 3 1 1 1 1 1];
%n = [5 5 3 2 2 2 2 2];
n = [6 6 6 5 5 5 4 4];

%%===================== Checking Input Errors =============================
if size(n)~= pcs,
    error('Quantization bit length does not match!');
end

%%===================== Reading image for processing ======================
%Img = double(imread('test_image2.tif', 1));
Img = double(imread(image, 1));
mval = max(max(max(Img)));
%Sc   = round((Img/mval)*256);

%%================== Breaking image into pattern vectors ==================
%D  = gha_chopstak(Sc,8,8)/256;

%[rP cP]=size(Img); 
[img_H img_W img_D]=size(Img);
r=floor(img_H/rm); 
c=floor(img_W/cm);
%D=zeros(r*c*img_D,p); 
%E=Img*0;
row=1;

for i=1:r, 
   for j=1:c,
       x=Img( (i-1)*rm+1:i*rm , (j-1)*cm+1:j*cm , :);
       x=x(:)';
       %row=(i-1)*r+j;
       %F(row,:)=[ (i-1)*rm+1 i*rm  (j-1)*cm+1 j*cm ];
       D(row,:)=x;
       row = row+1;
   end
end
AA = D;    %% original un-quantized vectors
D = D/256; %% pattern vectors that will input to the GHA

%%==================== GHA main routine ===================================
%W=gha(D,epochs,pcs,mu);

[rD cD]=size(D);
W=rand(pcs,cD)*(max(max(D))); % initial Weights

fprintf('Training in process...\n');
wh = waitbar(0,'Plotting...');

cc=cputime;
for epoch=1:epochs,
   ind=rand(1,rD); 
   [y inds]=sort(ind); 
   D = D(inds,:);
   for j=1:rD,
      x       = D(j,:); 
      Wd      = zeros(size(W));
      y(1)    = dot(W(1,:),x);
      Wd(1,:) = mu*y(1)*(x-y(1)*W(1,:));
      for h=2:pcs,
          y(h) = dot(W(h,:),x);
          temp = 0; 
          for k=1:h, 
              temp=temp+W(k,:)*y(k); 
          end
          Wd(h,:)=mu*y(h)*(x-temp);
      end
      W=W+Wd;
   end
   
   %% Store Weights for calculating MSE
   WW{epoch} = W';
   
   waitbar(epoch/epochs,wh);
   set(wh,'name',['Progress = ' sprintf('%2.1f',epoch/epochs*100) '%']);
   %fprintf(1,'Just trained epoch %g of %g. It took %g seconds.\n',epoch,epochs,etime(clock,cc))
end
close(wh);
fprintf('Training is done. It took %4.1f seconds\n',cputime-cc);
W=W';

%%======================== Get coefficient subroutine ======================
%coeffs=gha_getcoeffs(Img,W,1);

P    = AA;
mval = max(max(P));
P    = P/mval;
[rP cP] = size(P);      
[rW cW] = size(W);
coeffs  = zeros(rP,cW);
 
% first get the coeffs
for i=1:rP;  
    in=P(i,:)';   
    X=in(:,ones(1,cW));   
    coeffs(i,:)=sum(X.*W); 
    
    coeffs_other(i,:)=sum(X.*W_other);
end

PCAed=gha_recompose(coeffs,W,1,num_layer,r,c,rm,cm);

%%======= Displaying results ===================
figure;
%colormap(gray(256));
%set(gcf,'Position',[18   245   592   556])
subplot(3,2,1); 
image_empty(Img,num_layer);    
title('Original Image');
set(gca,'DataAspectRatio',[1 1 1]);

subplot(3,2,2); 
%gha_dispwe(W,200);
disp_num = 200;
W_norm = W-(min(min(W))); 
W_norm = W_norm/max(max(W_norm))*disp_num; 
%K=zeros(32,16);
for k=1:disp_w_row,
   for j=1:disp_w_col,
      K((k-1)*rm+(1:rm),(j-1)*cm+(1:cm), :)=reshape(W_norm(:,(k-1)*2+j),rm,cm,num_layer);
   end
end
image_empty(K,num_layer);
title('Weights');
set(gca,'DataAspectRatio',[1 1 1]);

subplot(3,2,3); 
image_empty(PCAed*256,num_layer);  % used to be PCAed*256   
title('Using First 8 Components');
set(gca,'DataAspectRatio',[1 1 1]);

%%====== Plotting quantization results
%[I, st, xla] = gha_quantcoeffs(coeffs,W,Img,n);
v  = std(coeffs).^2;     v_other = std(coeffs_other).^2;
b  = 2.^n;  
bpp= sum(n)/(rm*cm);
k  = sqrt(b./v);         k_other = sqrt(b./v_other);
% just multiplies each column by the corresponding value in k
M  = colmult(coeffs,k);  M_other  = colmult(coeffs,k_other);
% quantize
Q  = round(M);           Q_other = round(M_other);
% set upper and lower limits for thresholding the coefficients
toss=n-1; 
ul=2.^toss; 
ul(1)=2^n(1);
ll=-ul+1; ll(1)=1;
% threshold outliers
for i=1:pcs,
    if Q(:,i) < ll(i),
        Q(:,i) = ll(i);
    end
    if Q(:,i) > ul(i),
        Q(:,i) = ul(i);
    end           
end

for i=1:pcs,
    if Q_other(:,i) < ll(i),
        Q_other(:,i) = ll(i);
    end
    if Q_other(:,i) > ul(i),
        Q_other(:,i) = ul(i);
    end           
end

% convert from quantized values back to coefficients
H=colmult(Q,1./k);  H_other=colmult(Q_other,1./k_other);

%recompose image and display
I=gha_recompose(H,W,mval,num_layer,r,c,rm,cm);

mse=sqrt( sum(sum( sum((Img-I).^2))) /(rP*cP*num_layer));

st=['mse ' num2str(mse) '    bpp ' num2str(bpp)]
xla=num2str(n)

subplot(3,2,4);
image_empty(I,num_layer);     
title([num2str(round(8/bpp)),' to 1 compression']);
set(gca,'DataAspectRatio',[1 1 1]);


%recompose image using other weights and display
I_new=gha_recompose(H_other,W_other,mval_other,num_layer,r,c,rm,cm);

mse=sqrt( sum(sum( sum((Img-I_new).^2))) /(rP*cP*num_layer));

st=['mse ' num2str(mse) '    bpp ' num2str(bpp)]
xla=num2str(n)

subplot(3,2,5);
image_empty(I_new,num_layer);     
title([num2str(round(8/bpp)),' to 1 compression']);
set(gca,'DataAspectRatio',[1 1 1]);

subplot(3,2,6); 
%gha_dispwe(W,200);
disp_num = 200;
W_norm_new = W_other-(min(min(W_other))); 
W_norm_new = W_norm_new/max(max(W_norm_new))*disp_num; 
%K=zeros(32,16);
for k=1:disp_w_row,
   for j=1:disp_w_col,
      K_new((k-1)*rm+(1:rm),(j-1)*cm+(1:cm), :)=reshape(W_norm_new(:,(k-1)*2+j),rm,cm,num_layer);
   end
end
image_empty(K_new,num_layer);
title('Weights From Lena');
set(gca,'DataAspectRatio',[1 1 1]);

%calculating MSE
fprintf('Calculating MSE...\n');
for epoch = 1: epochs,
    II = gha_recompose(H,WW{epoch},mval,num_layer,r,c,rm,cm);
    MSE(epoch)=sqrt( sum(sum( sum((Img-II).^2))) /(rP*cP*num_layer));
end
figure;
plot(MSE);
xlabel('Number of epochs');
ylabel('MSE');
title('Learning curve');


function B=colmult(A, vec)
%% Small subroutine
cols=length(vec);
[rA cA]=size(A);
M=vec(ones(rA,1),:);
B=M.*A;
return


function image_empty(A,num_layer)
image(uint8(A));
if num_layer==1,
    colormap(gray(256));
end
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');
set(gca,'XTick',[]);
set(gca,'YTick',[]);
return

function I=gha_recompose(coeffs,W,mval,num_layer,r,c,rm,cm)
% I=recompose(coeffs,W,mval)
%
% coeffs - decomposition coefficents as determined via gha_getcoeffs.m
%
% mval scales the final output in the following line
%    I=gha_unchopst(imdat*mval, r, c, 8, 8);
% so if you want things to remain as normal, set mval = 1
%
% Hugh Pasika 1997

[rW, cW] = size(W);

% r  = 32;  %blocks per column 
% c  = 32;  %blocks per row
% rm = 8;   %rows/pixels in block
% cm = 8;   %columns/pixels in block

row = 1;
for i=1:r,
   for j=1:c,
      g  = coeffs(row,:)'; 
      CA = g(:,ones(1,rW))';
      CB = sum((CA.*W(:,:))')'; 
      imdat(row,:) = CB';
      row = row + 1;
   end
end

%I=gha_unchopst(imdat*mval, r, c, 8, 8);

%%=========== building image from unfolded blocks =========================
X = imdat*mval;

row = 1;
for i=1:r,
   for j=1:c,
      x = X(row, :);
      I( (i-1)*rm+1:i*rm, (j-1)*cm+1:j*cm, :) = reshape(x,rm,cm,num_layer);
      row = row + 1;
   end
end

return