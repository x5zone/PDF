function [obs, reliable] = observe(img, old_obs, object_color);
% OBSERVE Observes the object and returns the estimated observation vector. 
% Please note: this step is heavily simplified here compared to real-world
% computer vision tasks!
%
% INPUTS  : - im: Input image where the object has to be found.
%           - old_obs: Observation of the previous frame
%           - object_color: RGB color of the object.
% OUTPUTS : - obs: Estimated observation vector (position and velocity).
%           - reliable: True if a point has been found, false either.

if (nargin < 3)
    error('OBSERVE :: Not enough input arguments.');
end

% get binary image containing the object (by color)
segm_img = (img(:,:,1) == object_color(1) &...
           img(:,:,2) == object_color(2) &...
           img(:,:,3) == object_color(3));

% get coordinates
[val, y_ind] = max(segm_img);
[val, x_max] = max(max(segm_img));
y_max = y_ind(x_max);

% setup observation vector, including velocity
obs = [x_max; y_max; x_max - old_obs(1); y_max - old_obs(2)];

% check reliability
if obs == [1;1;0;0],
    % no color match has been found
    reliable = false;
else
    % color match found, everything good
    reliable = true;
end