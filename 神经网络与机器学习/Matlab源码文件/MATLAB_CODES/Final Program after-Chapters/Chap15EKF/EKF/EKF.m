function   [xkk,Pkk] = EKF(xkk,Pkk,inex,outex);

global  R lambda nwts ;

%==========================================================================
%predict
%==========================================================================

xkk1 = xkk;

Pkk1 = Pkk/lambda;

%==========================================================================
% correct
%==========================================================================

[zkk1,o1,o2] = MstEq(xkk1,inex);

resid = outex - zkk1;

H = findgrad(xkk1,inex,o1,o2,zkk1);

S =  H*Pkk1*H' + R;

G = Pkk1*H'/S;        

xkk = xkk1 + G*resid;

Pkk = (eye(nwts) - G*H)*Pkk1*(eye(nwts)-G*H)' + G*R*G';   

