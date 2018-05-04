function options = make_options(varargin)

% ML_OPTIONS - Generate/alter options structure for training classifiers
% ----------------------------------------------------------------------------------------%
%   options = ml_options('PARAM1',VALUE1,'PARAM2',VALUE2,...) 
%
%   Creates an options structure "options" in which the named parameters
%   have the specified values.  Any unspecified parameters are set to
%   default values specified below.
%   options = ml_options (with no input arguments) creates an options structure
%   "options" where all the fields are set to default values specified below.
%
%   Example: 
%          options=ml_options('Kernel','rbf','KernelParam',0.5,'NN',6);
%
%   "options" structure is as follows:
%
%            Field Name:  Description                                         : default 
% -------------------------------------------------------------------------------------
%              'Kernel':  'linear' | 'rbf' | 'poly'                           : 'linear'
%         'KernelParam':  --    | sigma | degree                              :  1 
%                  'NN':  number of nearest neighbor                          :  6
%'GraphDistanceFuncion':  distance function for graph: 'euclidean' | 'cosine' : 'euclidean' 
%        'GraphWeights':  'binary' | 'distance' | 'heat'                      : 'binary'
%    'GraphWeightParam':  e.g For heat kernel, width to use                   :  1 
%      'GraphNormalize':  Use normalized Graph laplacian (1) or not (0)       :  1
%          'ClassEdges':  Disconnect Edges across classes:yes(1) no (0)       :  0
%             'gamma_A':  RKHS norm regularization parameter (Ambient)        :  1 
%             'gamma_I':  Manifold regularization parameter  (Intrinsic)      :  1    
%                   mu':  Class-based Fully Supervised Laplacian Parameter    : 0.5
%   'LaplacianDegree'  :  Degree of Iterated Laplacian                        :  1  
%                   k  :  Regularized Spectral Representation Paramter        :  2
% -------------------------------------------------------------------------------------
%
% Acknowledgement: Adapted from Anton Schwaighofer's software:
%               http://www.cis.tugraz.at/igi/aschwaig/software.html
%
% Author: Vikas Sindhwani (vikass@cs.uchicago.edu)
% June 2004
% ----------------------------------------------------------------------------------------%

% options default values
options = struct('Kernel','linear', ...
                 'KernelParam',1, ...
                 'NN',6,...
                 'k' , 2, ...
                 'GraphDistanceFunction','euclidean',... 
                 'GraphWeights', 'binary', ...
                 'GraphWeightParam',1, ...
                 'GraphNormalize',1, ...
                 'ClassEdges',0,...
                 'LaplacianDegree',1, ...
                  'gamma_A',1.0,... 
                 'gamma_I',1.0, ...
                  'mu',0.5); 
  

numberargs = nargin;
if numberargs==0
    return;
end
    
Names = fieldnames(options);
[m,n] = size(Names);
names = lower(Names);

i = 1;
while i <= numberargs
  arg = varargin{i};
  if isstr(arg)
    break;
  end
  if ~isempty(arg)
    if ~isa(arg,'struct')
      error(sprintf('Expected argument %d to be a string parameter name or an options structure.', i));
    end
    for j = 1:m
      if any(strcmp(fieldnames(arg),Names{j,:}))
        val = getfield(arg, Names{j,:});
      else
        val = [];
      end
      if ~isempty(val)
        [valid, errmsg] = checkfield(Names{j,:},val);
        if valid
          options = setfield(options, Names{j,:},val);
        else
          error(errmsg);
        end
      end
    end
  end
  i = i + 1;
end

% A finite state machine to parse name-value pairs.
if rem(numberargs-i+1,2) ~= 0
  error('Arguments must occur in name-value pairs.');
end
expectval = 0;
while i <= numberargs
  arg = varargin{i};
  if ~expectval
    if ~isstr(arg)
      error(sprintf('Expected argument %d to be a string parameter name.', i));
    end
    lowArg = lower(arg);
    j = strmatch(lowArg,names);
    if isempty(j)
      error(sprintf('Unrecognized parameter name ''%s''.', arg));
    elseif length(j) > 1 
      % Check for any exact matches (in case any names are subsets of others)
      k = strmatch(lowArg,names,'exact');
      if length(k) == 1
        j = k;
      else
        msg = sprintf('Ambiguous parameter name ''%s'' ', arg);
        msg = [msg '(' Names{j(1),:}];
        for k = j(2:length(j))'
          msg = [msg ', ' Names{k,:}];
        end
        msg = sprintf('%s).', msg);
        error(msg);
      end
    end
    expectval = 1;
  else           
    [valid, errmsg] = checkfield(Names{j,:}, arg);
    if valid
      options = setfield(options, Names{j,:}, arg);
    else
      error(errmsg);
    end
    expectval = 0;
  end
  i = i + 1;
end
  

function [valid, errmsg] = checkfield(field,value)
% CHECKFIELD Check validity of structure field contents.
%   [VALID, MSG] = CHECKFIELD('field',V) checks the contents of the specified
%   value V to be valid for the field 'field'.
%

valid = 1;
errmsg = '';
if isempty(value)
  return
end
isFloat = length(value==1) & isa(value, 'double');
isPositive = isFloat & (value>=0);
isString = isa(value, 'char');
range = [];
requireInt = 0;
switch field
  case 'NN'
    requireInt = 1;
    range=[1 Inf];
case 'k'
    requireInt=1;
    range=[1 Inf];
 case 'GraphNormalize'
     requireInt = 1;
     range=[0 1];
 case 'LaplacianDegree'
     requireInt = 1;
     range=[1 Inf];
   case 'ClassEdges'
     requireInt = 1;
     range=[0 1];  
  case {'Kernel', 'GraphWeights', 'GraphDistanceFunction'}
    if ~isString,
      valid = 0;
      errmsg = sprintf('Invalid value for %s parameter: Must be a string', field);
    end
  case {'gamma_A', 'gamma_I','GraphWeightParam'}
    range = [0 Inf];
case {'mu'}
    range = [0 1];
  case 'KernelParam'
    valid = 1; 
  otherwise
    valid = 0;
    error('Unknown field name for Options structure.')
end

if ~isempty(range),
  if (value<range(1)) | (value>range(2)),
    valid = 0;
    errmsg = sprintf('Invalid value for %s parameter: Must be scalar in the range [%g..%g]', ...
                     field, range(1), range(2));
  end
end

if requireInt & ((value-round(value))~=0),
  valid = 0;
  errmsg = sprintf('Invalid value for %s parameter: Must be integer', ...
                   field);
end
