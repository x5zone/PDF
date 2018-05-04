function [im, x, y] = simulation(t, object_color, bar_width);
% SIMULATION Simulates the trajectory as an image sequence.
% It shows the object if it moves over background and hides it if occluded
% by foreground areas.
%
% INPUTS  : - t: The frame (time) to observe.
%           - object_color: RGB color of the object.
%           - bar_width: The width of the foreground bars.
% OUTPUTS : - im: The image containing the object as a square.
%           - x: The x coordinate of the object for evaluation.
%           - y: The y coordinate of the object for evaluation.

if(nargin < 3)
    bar_width = 10;
end
if(nargin < 2)
    error('SIMULATION :: Not enough input arguments.');
end

% define constants
height = 300;
width = 300;
bar_color = [0 0 0];

% calculate position of the object
x = floor(50 + 200*abs(sin(2*pi*t)));
y = floor(150 + 100*sin(3.5*pi*t));

% create blank image, do not paint object at time 0 (background only)
im = ones(height,width,3);

if(t>0)
    % paint object
    for i=1:3,
        im(y+(-2:2),x+(-2:2),i) = object_color(i);
    end
end

% paint foreground (maybe on top of object)
for bar = (width/8-bar_width/2):(width/4):width, 
    for i=1:3,
        im(floor(bar)+(0:bar_width), :, i) = bar_color(i);
    end
end
 