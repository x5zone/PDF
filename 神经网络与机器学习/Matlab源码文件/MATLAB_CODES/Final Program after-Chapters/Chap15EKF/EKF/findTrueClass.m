
function outdata = findTrueClass(indata)

global r1 r2 r3;

N = size(indata,2);

outdata = [];

for i = 1:N
    
    inex = indata(:,i);
    
    d = norm(inex);
    
    if  (d <= r3)        
       
        if ( (d< r1) && inex(2)>0 ) || ( (d< r2) && (d>r1) &&  inex(2)<0 ) || ( (d > r2) && inex(2)>0 ) 
            outex = 1;
        else
            outex = -1;
        end;
        
        outdata = [outdata outex];
        
    end;
    
end;
