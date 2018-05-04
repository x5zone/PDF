
function [Y,o1,o2] = MstEq(W,inex)

global nhidden1 nhidden2  nin nout c1 c2;

W1 = reshape( W(1:nhidden1*(nin+1)),nhidden1,nin+1);
 
W2 = reshape( W(nhidden1*(nin+1)+1:nhidden1*(nin+1)+nhidden2*(nhidden1+1)),nhidden2,nhidden1+1);
   
W3 = reshape( W(nhidden1*(nin+1)+nhidden2*(nhidden1+1)+1:end),nout,nhidden2+1);
    
u1 = W1*[inex;1];

o1 = c1*( (2./(1+exp(-2*c2*u1))) - 1);
    
u2 = W2*[o1;1];   
    
o2 = c1*( (2./(1+exp(-2*c2*u2))) - 1);
    
u3 = W3*[o2;1];   
    
Y  = c1*( (2./(1+exp(-2*c2*u3))) - 1);      
    

