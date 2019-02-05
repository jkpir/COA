import numpy as np

def COA(FOBJ, lu, nfevalMAX, n_packs=20, n_coy=5):

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

    # Optimization problem variables
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]

    # Algorithm parameters
    if n_coy < 3:
        raise Exception("At least 3 coyotes per pack must be used")

    # Probability of leaving a pack
    p_leave = 0.005*(n_coy**2)
    Ps = 1/D

    # Packs initialization (Eq. 2)
    pop_total = n_packs*n_coy
    costs = np.zeros((1, pop_total))
    coyotes = np.tile(VarMin, [pop_total, 1]) + np.random.rand(pop_total, D) * np.tile(VarMax, [pop_total, 1]) - \
              np.tile(VarMin, [pop_total, 1])
    ages = np.zeros((1, pop_total))
    packs = np.random.permutation(pop_total).reshape(n_packs, n_coy)

    # Evaluate coyotes adaptation (Eq. 3)
    for c in range(pop_total):
        costs[0, c] = FOBJ(coyotes[c, :])

    nfeval = pop_total

    # Output variables
    globalMin = np.min(costs[0, :])
    ibest = np.argmin(costs[0, :])
    globalParams = coyotes[ibest, :]

    # Main loop
    year = 1
    while nfeval < nfevalMAX:  # Stopping criteria
        # Update the years counter
        year += 1

        # Execute the operations inside each pack
        for p in range(n_packs):
            # Get the coyotes that belong to each pack
            coyotes_aux = coyotes[packs[p, :], :]
            costs_aux = costs[0, packs[p, :]]
            ages_aux = ages[0, packs[p, :]]

            # Detect alphas according to the costs (Eq. 5)
            ind = np.argsort(costs_aux)
            costs_aux = costs_aux[ind]
            coyotes_aux = coyotes_aux[ind, :]
            ages_aux = ages_aux[ind]
            c_alpha = coyotes_aux[0, :]

            # Compute the social tendency of the pack (Eq. 6)
            tendency = np.median(coyotes_aux, 0)

            #  Update coyotes' social condition
            new_coyotes = np.zeros((n_coy, D))
            for c in range(n_coy):
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(n_coy)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(n_coy)

                # Try to update the social condition according
                # to the alpha and the pack tendency(Eq. 12)
                new_coyotes[c, :] = coyotes_aux[c, :] + np.random.rand()*(c_alpha - coyotes_aux[rc1, :]) + \
                                    np.random.rand()*(tendency - coyotes_aux[rc2, :])

                # Keep the coyotes in the search space (optimization problem constraint)
                new_coyotes[c, :] = Limita(new_coyotes[c, :], D, VarMin, VarMax)

                # Evaluate the new social condition (Eq. 13)
                new_cost = FOBJ(new_coyotes[c, :])
                nfeval += 1

                # Adaptation (Eq. 14)
                if new_cost < costs_aux[c]:
                    costs_aux[c] = new_cost
                    coyotes_aux[c, :] = new_coyotes[c, :]

            # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            parents = np.random.permutation(n_coy)[:2]
            prob1 = (1-Ps)/2
            prob2 = prob1
            pdr = np.random.permutation(D)
            p1 = np.zeros((1, D))
            p2 = np.zeros((1, D))
            p1[0, pdr[0]] = 1  # Guarantee 1 charac. per individual
            p2[0, pdr[1]] = 1  # Guarantee 1 charac. per individual
            r = np.random.rand(1, D-2)
            p1[0, pdr[2:]] = r < prob1
            p2[0, pdr[2:]] = r > 1-prob2

            # Eventual noise
            n = np.logical_not(np.logical_or(p1, p2))

            # Generate the pup considering intrinsic and extrinsic influence
            pup = p1*coyotes_aux[parents[0], :] + \
                  p2*coyotes_aux[parents[1], :] + \
                  n*(VarMin + np.random.rand(1, D) * (VarMax - VarMin))

            # Verify if the pup will survive
            pup_cost = FOBJ(pup[0, :])
            nfeval += 1
            worst = np.flatnonzero(costs_aux > pup_cost)
            if len(worst) > 0:
                older = np.argsort(ages_aux[worst])
                which = worst[older[::-1]]
                coyotes_aux[which[0], :] = pup
                costs_aux[which[0]] = pup_cost
                ages_aux[which[0]] = 0

            # Update the pack information
            coyotes[packs[p], :] = coyotes_aux
            costs[0, packs[p]] = costs_aux
            ages[0, packs[p]] = ages_aux

        # A coyote can leave a pack and enter in another pack (Eq. 4)
        if n_packs > 1:
            if np.random.rand() < p_leave:
                rp = np.random.permutation(n_packs)[:2]
                rc = [np.random.randint(0, n_coy), np.random.randint(0, n_coy)]
                aux = packs[rp[0], rc[0]]
                packs[rp[0], rc[0]] = packs[rp[1], rc[1]]
                packs[rp[1], rc[1]] = aux

        # Update coyotes ages
        ages += 1

        # Output variables (best alpha coyote among all alphas)
        globalMin = np.min(costs[0, :])
        ibest = np.argmin(costs)
        globalParams = coyotes[ibest, :]

    return globalMin, globalParams

def Limita(X, D, VarMin, VarMax):
    # Keep the coyotes in the search space (optimization problem constraint)
    for abc in range(D):
        X[abc] = max([min([X[abc], VarMax[abc]]), VarMin[abc]])

    return X

def Sphere(X):
    y = np.sum(X**2)

    return y

if __name__=="__main__":

    import time
    # Objective function definition
    fobj = Sphere           # Function
    d = 10                  # Problem dimension
    lu = np.zeros((2, d))   # Boundaires
    lu[0, :] = -10          # Lower boundaires
    lu[1, :] = 10           # Upper boundaries

    # COA parameters
    n_packs = 20            # Number of Packs
    n_coy = 5               # Number of coyotes
    nfevalmax = 20000       # Stopping criteria: maximum number of function evaluations

    # Experimanetal variables
    n_exper = 3             # Number of experiments
    t = time.time()         # Time counter (and initial value)
    y = np.zeros(n_exper)   # Experiments costs (for stats.)
    for i in range(n_exper):
        # Apply the COA to the problem with the defined parameters
        gbest, par = COA(fobj, lu, nfevalmax, n_packs, n_coy)
        # Keep the global best
        y[i] = gbest
        # Show the result (objective cost and time)
        print("Experiment ", i+1, ", Best: ", gbest, ", time (s): ", time.time()-t)
        t = time.time()

    # Show the statistics
    print("Statistics (min., avg., median, max., std.)")
    print([np.min(y), np.mean(y), np.median(y), np.max(y), np.std(y)])