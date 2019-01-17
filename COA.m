function [GlobalParams,GlobalMin] = COA(FOBJ, lu, nfevalMAX,n_packs,n_coy)
%% ------------------------------------------------------------------------
% Coyote Optimization Algorithm (COA) for Global Optimization.
% A nature-inspired metaheuristic proposed by Juliano Pierezan and 
% Leandro dos Santos Coelho (2018).
%
% Pierezan, J. and Coelho, L. S. "Coyote Optimization Algorithm: A new 
% metaheuristic for global optimization problems", Proceedings of the IEEE 
% Congress on Evolutionary Computation (CEC), Rio de Janeiro, Brazil, July 
% 2018, pages 2633-2640.
%
% Example:
% FOBJ        = @(x) sum(x.^2);         % Optimization problem
% D           = 30;                     % Problem dimension
% lu          = [zeros(1,D);ones(1,D)]; % Seach space
% nfevalMAX   = 10000*D;                % Stopping criteria
% Np          = 10;                     % Number of packs
% Nc          = 10;                     % Number of coyotes
% [GlobalParams,GlobalMin] = COA(FOBJ, lu, nfevalMAX,Np,Nc);
%
% Federal University of Parana (UFPR), Curitiba, Parana, Brazil.
% juliano.pierezan@ufpr.br
%% ------------------------------------------------------------------------

%% Optimization problem variables
D           = size(lu,2);
VarMin      = lu(1,:);
VarMax      = lu(2,:);

%% Algorithm parameters
if nargin < 5, n_coy = 5; 
elseif n_coy < 3, error('At least 3 coyotes per pack!'); end
if nargin < 4, n_packs = 20; end

% Probability of leaving a pack
p_leave     = 0.005*n_coy^2;
Ps          = 1/D;

%% Packs initialization (Eq. 2)
pop_total   = n_packs*n_coy;
costs       = zeros(pop_total,1);
coyotes     = repmat(VarMin,pop_total,1) + rand(pop_total,D).*(repmat(VarMax,pop_total,1) - repmat(VarMin,pop_total,1));
ages        = zeros(pop_total,1);
packs       = reshape(randperm(pop_total),n_packs,[]);
coypack     = repmat(n_coy,n_packs,1);

%% Evaluate coyotes adaptation (Eq. 3)
for c=1:pop_total
    costs(c,1) = FOBJ(coyotes(c,:));
end
nfeval = pop_total;

%% Output variables
[GlobalMin,ibest]   = min(costs);
GlobalParams        = coyotes(ibest,:); 
    
%% Main loop
year=0;
while nfeval<nfevalMAX % Stopping criteria
    
    %% Update the years counter
    year = year + 1;

    %% Execute the operations inside each pack
    for p=1:n_packs
        % Get the coyotes that belong to each pack
        coyotes_aux = coyotes(packs(p,:),:);
        costs_aux   = costs(packs(p,:),:);
        ages_aux    = ages(packs(p,:),1);
        n_coy_aux   = coypack(p,1);
        
        % Detect alphas according to the costs (Eq. 5)
        [costs_aux,inds] = sort(costs_aux,'ascend');
        coyotes_aux      = coyotes_aux(inds,:);
        ages_aux         = ages_aux(inds,:);
        c_alpha          = coyotes_aux(1,:);
        
        % Compute the social tendency of the pack (Eq. 6)
        tendency         = median(coyotes_aux,1);
        
        % Update coyotes' social condition
        new_coyotes      = zeros(n_coy_aux,D);
        for c=1:n_coy_aux
            rc1 = c;
            while rc1==c
                rc1 = randi(n_coy_aux);
            end
            rc2 = c;
            while rc2==c || rc2 == rc1
                rc2 = randi(n_coy_aux);
            end
            
            % Try to update the social condition according to the alpha and
            % the pack tendency (Eq. 12)
            new_c = coyotes_aux(c,:) + rand*(c_alpha - coyotes_aux(rc1,:))+ ...
                                       rand*(tendency  - coyotes_aux(rc2,:));
            
            % Keep the coyotes in the search space (optimization problem
            % constraint)
            new_coyotes(c,:) = min(max(new_c,VarMin),VarMax);
            
            % Evaluate the new social condition (Eq. 13)
            new_cost = FOBJ(new_coyotes(c,:));
            nfeval   = nfeval+1;
            
            % Adaptation (Eq. 14)
            if new_cost < costs_aux(c,1)
                costs_aux(c,1)      = new_cost;
                coyotes_aux(c,:)    = new_coyotes(c,:);
            end
        end
        
        %% Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
        parents         = randperm(n_coy_aux,2);
        prob1           = (1-Ps)/2;
        prob2           = prob1;
        pdr             = randperm(D);
        p1              = zeros(1,D);
        p2              = zeros(1,D);
        p1(pdr(1))      = 1; % Guarantee 1 charac. per individual
        p2(pdr(2))      = 1; % Guarantee 1 charac. per individual
        r               = rand(1,D-2);
        p1(pdr(3:end))  = r < prob1;
        p2(pdr(3:end))  = r > 1-prob2;
        
        % Eventual noise 
        n  = ~(p1|p2);
        
        % Generate the pup considering intrinsic and extrinsic influence
        pup =   p1.*coyotes_aux(parents(1),:) + ...
                p2.*coyotes_aux(parents(2),:) + ...
                n.*(VarMin + rand(1,D).*(VarMax-VarMin));
        
        % Verify if the pup will survive
        pup_cost    = FOBJ(pup);
        nfeval      = nfeval + 1;
        worst       = find(pup_cost<costs_aux==1);
        if ~isempty(worst)
            [~,older]               = sort(ages_aux(worst),'descend');
            which                   = worst(older);
            coyotes_aux(which(1),:) = pup;
            costs_aux(which(1),1)   = pup_cost;
            ages_aux(which(1),1)    = 0;
        end
        
        %% Update the pack information
        coyotes(packs(p,:),:) = coyotes_aux;
        costs(packs(p,:),:)   = costs_aux;
        ages(packs(p,:),1)    = ages_aux;
    end
    
    %% A coyote can leave a pack and enter in another pack (Eq. 4)
    if n_packs>1
        if rand < p_leave
            rp                  = randperm(n_packs,2);
            rc                  = [randperm(coypack(rp(1),1),1) ...
                                   randperm(coypack(rp(2),1),1)];
            aux                 = packs(rp(1),rc(1));
            packs(rp(1),rc(1))  = packs(rp(2),rc(2));
            packs(rp(2),rc(2))  = aux;
        end
    end
    
    %% Update coyotes ages
    ages = ages + 1;
    
    %% Output variables (best alpha coyote among all alphas)
    [GlobalMin,ibest]   = min(costs);
    GlobalParams        = coyotes(ibest,:);    
end
end
