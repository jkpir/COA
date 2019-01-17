clear all
close all
clc

% Optimization setup
FOBJ        = @(x) Rastrigin(x);      % Optimization problem
D           = 30;                     % Problem dimension
lu          = [zeros(1,D);ones(1,D)]; % Seach space
nfevalMAX   = 10000*D;                % Stopping criteria
Np          = 10;                     % Number of packs
Nc          = 10;                     % Number of coyotes
[GlobalParams,GlobalMin] = COA(FOBJ, lu, nfevalMAX,Np,Nc); % Start process

% Show results
fprintf(1,'COA''s global optimum: %.4g\n',GlobalMin);
for i=1:D
    fprintf(1,'COA''s global parameters are x(%d): %.4f\n',i,GlobalParams(i));
end
