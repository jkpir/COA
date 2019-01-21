''' ------------------------------------------------------------------------
% Coyote Optimization Algorithm (COA) for Global Optimization.
% A nature-inspired metaheuristic proposed by Juliano Pierezan and 
% Leandro dos Santos Coelho (2018).
%
% Pierezan, J. and Coelho, L. S. "Coyote Optimization Algorithm: A new 
% metaheuristic for global optimization problems", Proceedings of the IEEE 
% Congress on Evolutionary Computation (CEC), Rio de Janeiro, Brazil, July 
% 2018, pages 2633-2640.
%
% Federal University of Parana (UFPR), Curitiba, Parana, Brazil.
% juliano.pierezan@ufpr.br
% 
% Python version coded by Rodrigo Meira de Andrade (Jan-2019) @rodrigo2019
%
%% ---------------------------------------------------------------------'''

import numpy as np

def COA(FOBJ, lu, nfevalMAX, n_packs=20, n_coy=5):
"""
Inputs: 
FOBJ: objective function
lu: Lower and upper boundaries
nfevalMAX: maximum number of function evaluations
n_packs: number of packs
n_coys: number of coyotes

Outputs:
GlobalMin: minimum global cost achieved
GlobalParams: set of optimized variables
"""

    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]

    if n_coy < 3:
        raise Exception("At least 3 coyotes per pack must be used")

    p_leave = 0.005*n_coy**2
    Ps = 1/D

    pop_total = n_packs*n_coy
    costs = np.zeros((pop_total, 1))
    coyotes = np.tile(VarMin, [pop_total, 1]) + np.random.rand(pop_total, D) * np.tile(VarMax, [pop_total, 1]) - \
              np.tile(VarMin, [pop_total, 1])
    ages = np.zeros((pop_total, 1))
    packs = np.random.permutation(pop_total).reshape(n_packs, -1)
    coypack = np.tile(n_coy, [n_packs, 1])

    for c in range(pop_total):
        costs[c, 0] = FOBJ(coyotes[c, :])
    nfeval = pop_total
    GlobalMin = min(costs)
    ibest = costs.argmin()
    GlobalParams = coyotes[ibest]

    year = 1
    while nfeval < nfevalMAX:
        year += 1
        for p in range(n_packs):
            coyotes_aux = coyotes[packs[p, :], :]
            costs_aux = costs[packs[p, :], :]
            ages_aux = ages[packs[p, :], :]
            n_coy_aux = coypack[p, 0]

            inds = costs_aux[:, 0].argsort()
            costs_aux.sort()
            coyotes_aux = coyotes_aux[inds]
            ages_aux = ages_aux[inds]
            c_alpha = coyotes_aux[0]

            tendency = np.median(coyotes_aux)

            new_coyotes = np.zeros((n_coy, D))

            for c in range(n_coy_aux):
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(n_coy_aux)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(n_coy_aux)

                new_c = coyotes_aux[c] + np.random.rand()*(c_alpha - coyotes_aux[rc1]) + np.random.rand()*(tendency - coyotes_aux[rc2])
                new_coyotes[c, 0] = min(max(new_c.max(), VarMin.max()), VarMax.min())

                new_cost = FOBJ(new_coyotes[c])
                nfeval += 1

                if new_cost < costs[c, 0]:
                    costs_aux[c, 0] = new_cost
                    coyotes_aux[c] = new_coyotes[c]

            parents = np.random.permutation(n_coy_aux)[:2]
            prob1 = (1-Ps)/2
            prob2 = prob1
            pdr = np.random.permutation(D)
            p1 = np.zeros((1, D))
            p2 = np.zeros((1, D))
            p1[0, pdr[0]] = 1
            p2[0, pdr[1]] = 1
            r = np.random.rand(1, D-2)
            p1[0, pdr[2:]] = r < prob1
            p2[0, pdr[2:]] = r < 1-prob2

            n = np.logical_not(np.logical_or(p1, p2))

            pup = p1*coyotes_aux[parents[0]] + p2*coyotes_aux[parents[1]] + n*(VarMin + np.random.rand(1, D) * (VarMax -
                                                                                                                VarMin))

            pup_cost = FOBJ(pup[0])
            nfeval += 1
            worst = np.flatnonzero(pup_cost < costs_aux)
            if len(worst) > 0:
                older = worst.argsort()
                older = older[::-1]
                which = worst[older]
                coyotes_aux[which[0], 0] = pup_cost
                ages_aux[which[0], 0] = 0

            coyotes[packs[p]] = coyotes_aux
            costs[packs[p]] = costs_aux
            ages[packs[p]] = ages_aux

            if n_packs > 1:
                if np.random.rand() < p_leave:
                    rp = np.random.permutation(n_packs)[:2]
                    rc = [np.random.permutation(coypack[rp[0], 0])[0], np.random.permutation(coypack[rp[1], 0])[0]]
                    rc = np.asarray(rc)
                    aux = packs[rp[0], rc[0]]
                    packs[rp[0], rc[0]] = packs[rp[1], rc[1]]
                    packs[rp[1], rc[1]] = aux

            ages += 1

            GlobalMin = costs.min()
            GlobalParams = coyotes[costs.argmin()]

    return GlobalMin, GlobalParams


if __name__=="__main__":

    import time
    f = lambda x: sum(x**2)
    d = 30
    lu = np.zeros((2, d))
    lu[1, :] = 1
    nfeval = 1000*d
    Np = 10
    Nc = 10
    t = time.time()
    for i in range(10):
        mini, par = COA(f, lu, nfeval)
        print(time.time()-t)
        t = time.time()
    print(mini, par)


