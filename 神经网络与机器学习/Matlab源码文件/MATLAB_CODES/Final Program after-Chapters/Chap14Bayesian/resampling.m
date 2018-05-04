function outIndex = resampling(inIndex,wn);
% RESAMPLING Applies the systematic resampling algorithm to the input.
% INPUTS  : - inIndex: Input particle indices
%           - wn: Normalized importance ratios
% OUTPUTS : - outIndex: Resampled indices
% NOTE    : Thanks to Arnaud Doucet and Nando de Freitas.

if (nargin < 2)
    error('RESAMPLING :: Not enough input arguments.');
end

% Initialisation
wn=wn';
[arb,N] = size(wn);  % N = Number of particles.
N_children=zeros(1,N);
label=zeros(1,N);
label=1:1:N;
s=1/N;
auxw=0;
auxl=0;
li=0;   % Label of the current point
T=s*rand(1);
j=1;
Q=0;
i=0;

% Sampling before
u=rand(1,N);
while (T<1)
   if (Q>T)
      T=T+s;
      N_children(1,li)=N_children(1,li)+1;
   else
      % select i uniformly between j and N
      i=fix((N-j+1)*u(1,j))+j;
      % save the associate characteristic
      auxw=wn(1,i);
      li=label(1,i);
      % update the cfd
      Q=Q+auxw;
      % swap 
      wn(1,i)=wn(1,j);
      label(1,i)=label(1,j);
      %wn(1,j)=auxw;
      %label(1,j)=li;
      j=j+1;
   end
end

% Copy resampled trajectories
index=1;
for i=1:N
  if (N_children(1,i)>0)
    for j=index:index+N_children(1,i)-1
      outIndex(j) = inIndex(i);
    end;
  end;   
  index= index+N_children(1,i);   
end