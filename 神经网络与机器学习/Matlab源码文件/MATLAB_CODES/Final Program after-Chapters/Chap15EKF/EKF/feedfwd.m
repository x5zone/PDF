function YArray = feedfwd(W,inex)

global nhidden1 nhidden2 nin nout c1 c2;

W1 = reshape( W(1:nhidden1*(nin+1)),nhidden1,nin+1);
 
W2 = reshape( W(nhidden1*(nin+1)+1:nhidden1*(nin+1)+nhidden2*(nhidden1+1)),nhidden2,nhidden1+1);
   
W3 = reshape( W(nhidden1*(nin+1)+nhidden2*(nhidden1+1)+1:end),nout,nhidden2+1);

L = size(inex,2);

u1 = W1*[inex;ones(1,L)];

o1 = c1*( (2./(1+exp(-2*c2*u1))) - 1);
    
u2 = W2*[o1;ones(1,L)];   
    
o2 = c1*( (2./(1+exp(-2*c2*u2))) - 1);
    
u3 = W3*[o2;ones(1,L)];   
    
YArray  = c1*( (2./(1+exp(-2*c2*u3))) - 1);     
    
    

