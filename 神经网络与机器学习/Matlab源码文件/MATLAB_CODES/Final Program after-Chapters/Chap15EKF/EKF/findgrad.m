function dCdW = findgrad(W,inex,o1,o2,o3);

global nwts nin nhidden1 nhidden2 nout c1 c2;

W2 = reshape( W(nhidden1*(nin+1)+1:nhidden1*(nin+1)+nhidden2*(nhidden1+1)),nhidden2,nhidden1+1);
   
W3 = reshape( W(nhidden1*(nin+1)+nhidden2*(nhidden1+1)+1:end),nout,nhidden2+1);

%==========================================================================

delta3 = (c2/c1)*(c1^2 - o3.^2);

delta2 = (c2/c1)*(c1^2 - o2.^2).*( [W3(:,1:nhidden2)]'*delta3 );

delta1 = (c2/c1)*(c1^2 - o1.^2).*( [W2(:,1:nhidden1)]'*delta2 );

%==========================================================================

dCdW1 = delta1*[inex;1]';

dCdW2 = delta2*[o1;1]';

dCdW3 = delta3*[o2;1]';

dCdW = [dCdW1(:);  dCdW2(:); dCdW3(:) ]';
