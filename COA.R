# -- COA algorithm
COA <- function(fobj,lu,nfevalMAX,n_packs,n_coy){
# ------------------------------------------------------------------------
# Coyote Optimization Algorithm (COA) for Global Optimization.
# A nature-inspired metaheuristic proposed by Juliano Pierezan and
# Leandro dos Santos Coelho (2018).
#
# Pierezan, J. and Coelho, L. S. "Coyote Optimization Algorithm: A new
# metaheuristic for global optimization problems", Proceedings of the IEEE
# Congress on Evolutionary Computation (CEC), Rio de Janeiro, Brazil, July
# 2018, pages 2633-2640.
#
# Federal University of Parana (UFPR), Curitiba, Parana, Brazil.
# juliano.pierezan@ufpr.br
# ------------------------------------------------------------------------

# Input varibles
VarMin = lu[1,]
VarMax = lu[2,]
D = length(VarMin)

# Probability of leaving a pack
p_leave     = 0.005*n_coy^2;
Ps          = 1/D;

# Algorithm variables
pop_total 	= n_packs*n_coy
coyotes 	= matrix(0,pop_total,D)
costs   	= matrix(0,pop_total,1)
packs 	= matrix(0,n_packs,n_coy)
ages        = matrix(0,pop_total,1)

# Separate the packs
RC = sample(1:pop_total, pop_total, replace=F)
for (i in 1:n_packs){
packs[i,] = RC[((i-1)*n_coy+1) : (i*n_coy)]
}

# Initial population
for (i in 1:pop_total){
	coyotes[i,] = VarMin + runif(D,0,1)*(VarMax - VarMin)
	costs[i,] = fobj(coyotes[i,])
}
nfeval = pop_total

# Main loop
year=0;
while(nfeval < nfevalMAX){

# Update the years counter
year = year + 1

# Execute the operations inside each pack
for (p in 1:n_packs) {

        # Get the coyotes that belong to each pack
        coyotes_aux = coyotes[packs[p,],]
        costs_aux   = costs[packs[p,],]
        ages_aux    = ages[packs[p,],1]

	  # Detect alphas according to the costs (Eq. 5)
        s 			       = sort(costs_aux, decreasing=FALSE, index.return=TRUE, )
	  costs_aux 	 	 = s$x
	  inds 		 	 = s$ix	
        coyotes_aux[1:n_coy,]  = coyotes_aux[inds,]
        ages_aux         	 = ages_aux[inds]
        c_alpha          	 = coyotes_aux[1,]

	  # Compute the social tendency of the pack (Eq. 6)
        tendency         	 = apply(coyotes_aux, 2, FUN = median)
        
        # Update coyotes' social condition
        new_coyotes      	 = matrix(0,n_coy,D)
	  for (c in 1:n_coy) {
            rc1 = c
            while (rc1==c){
                rc1 = sample(n_coy,1);
            }
		rc2 = c
		while ((rc2==c) | (rc2==rc1)){
                rc2 = sample(n_coy,1);
            }
            
            # Try to update the social condition according to the alpha and
            # the pack tendency (Eq. 12)
            new_coyotes[c,] = (coyotes_aux[c,] + 
					runif(1,0,1)*(c_alpha - coyotes_aux[rc1,]) + 
					runif(1,0,1)*(tendency  - coyotes_aux[rc2,]))
            
            # Keep the coyotes in the search space (optimization problem
            # constraint)
		for (aux in 1:D){
	      	new_coyotes[aux] = max(new_coyotes[aux],VarMin[aux])
			new_coyotes[aux] = min(new_coyotes[aux],VarMax[aux])
		}
            
            # Evaluate the new social condition (Eq. 13)
            new_cost = fobj(new_coyotes[c,]);
            nfeval   = nfeval+1;
            
            # Adaptation (Eq. 14)
            if (new_cost < costs_aux[c]){
                costs_aux[c]   	= new_cost
                coyotes_aux[c,]  	= new_coyotes[c,]
            }
         } # END Coyotes Loop
	
	  # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
        parents         = sample(n_coy,2,replace=F)
        prob1           = (1-Ps)/2
        prob2           = prob1;
        pdr             = sample(D,D,replace=F)
        p1              = matrix(0,1,D)
        p2              = matrix(0,1,D)
        p1[pdr[1]]      = 1 # Guarantee 1 charac. per individual
        p2[pdr[2]]      = 1 # Guarantee 1 charac. per individual
        r               = runif(D-2,0,1)
        p1[pdr[3:D]]    = r < prob1
        p2[pdr[3:D]]    = r > 1-prob2;

	  # Eventual noise 
        n  = !(p1|p2);
        
        # Generate the pup considering intrinsic and extrinsic influence
        pup =   (p1*coyotes_aux[parents[1],] + 
		     p2*coyotes_aux[parents[2],] + 
		     n*(VarMin + runif(D,0,1)*(VarMax-VarMin)))
        
        # Verify if the pup will survive
        pup_cost    = fobj(pup)
        nfeval      = nfeval + 1
        worst       = which(pup_cost < costs_aux, arr.ind = FALSE)
        n_worst     = length(worst)
        if (n_worst>0){
		s 			  	= sort(ages_aux[worst], decreasing=TRUE, index.return=TRUE)
	      older		 	 	= s$ix
		older 			= older[1]
            qual                   	= worst[older]
            coyotes_aux[qual[1],] 	= pup
            costs_aux[qual[1]]   	= pup_cost
            ages_aux[qual[1]]    	= 0
        }
 
        # Update the pack information
        coyotes[packs[p,],] = coyotes_aux;
        costs[packs[p,],]   = costs_aux;
        ages[packs[p,],]    = ages_aux;

	} # END Packs Loop

    # A coyote can leave a pack and enter in another pack (Eq. 4)
    if (n_packs>1){
        if (runif(1,0,1) < p_leave){
            rp                  = sample(n_packs,2,replace=F)
            rc                  = sample(n_coy,2,replace=T)
            aux                 = packs[rp[1],rc[1]]
            packs[rp[1],rc[1]]  = packs[rp[2],rc[2]]
            packs[rp[2],rc[2]]  = aux
        }
    }


# -- Update coyotes ages
ages = ages + 1;

# -- Outputs
globalMin = min(costs)
ibest = which.min(costs)
globalParams = coyotes[ibest,]

} # END Main Loop

result <- list(globalMin=globalMin, globalParams=globalParams)

return(result)
} # END Coyote Optimization Algorithm

## ----------------------------------------------------------
#  Example:

# Objective function parameters
fobj <- function(x){sum(x**2)}
D = 10
lu = matrix(-10,2,D)
lu[2,] = 10

# -- COA parameters
n_packs = 20
n_coy = 5
nfevalMAX = 20000

# -- Run the experiments
n_exper = 3; 	 # Number of experiments
y = matrix(0,1,n_exper)
t <- Sys.time()
for (i in 1:n_exper){
result = COA(fobj,lu,nfevalMAX,n_packs,n_coy)
print(Sys.time() - t)
y[1,i] = result$globalMin
t <- Sys.time()
}

# -- Show the statistics
v = c(min(y), mean(y), median(y), max(y), sd(y))
print('Statistics (min., avg., median, max., std.)')
print(v)