clear all
close all
clc

% Objective function setup
FOBJ        = @(x) sum(x.^2);        % Optimization problem
D           = 10;                    % Problem dimension
lu          = [-10*ones(1,D);
                10*ones(1,D)];       % Seach space
% COA paramters setup         
nfevalMAX   = 20000;                 % Stopping criteria
Np          = 20;                    % Number of packs
Nc          = 5;                     % Number of coyotes

% Experimental setup
n_exper = 3;                         % Number of experiments
y = zeros(1,n_exper);                % Objective costs achieved
t = clock();                         % Time counter (initial value)
for i=1:n_exper
    % Apply the COA to the optimization problem
    [~,y(1,i)] = COA(FOBJ, lu, nfevalMAX,Np,Nc); % Start process
    % Show the result (cost and time)
    fprintf(1,'\nExperiment : %d, cost: %.6f, time: %.4f',...
        i,y(1,i),etime(clock, t));
    % Update time counter
    t = clock();
end
    