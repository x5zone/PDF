function [new_state] = process_model(state);
% PROCESS_MODEL Models the movement of the object.
% The assumption in this model is that the velocity is constant.
%
% INPUTS  : - state:  Current state.
% OUTPUTS : - new_state: Predicted state.

if (nargin < 1)
    error('PROCESS_MODEL :: Not enough input arguments.');
end

A = [1.0 0.0 1.0 0.0; ...  % x_t = x_t-1 + vx_t-1
     0.0 1.0 0.0 1.0; ...  % y_t = y_t-1 + vy_t-1
     0.0 0.0 1.0 0.0; ...  % vx_t = vx_t-1
     0.0 0.0 0.0 1.0];     % vy_t = vy_t-1    
     
new_state = A * state;







