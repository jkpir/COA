import numpy as np

def COA(FOBJ, lu, nfevalMAX, n_packs=20, n_coy=5):

    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]

    if n_coy < 3:
        raise Exception("At least 3 coyotes per pack must be used")

    p_leave = 0.005*(n_coy**2)
    Ps = 1/D

    pop_total = n_packs*n_coy
    costs = np.zeros((1, pop_total))
    coyotes = np.tile(VarMin, [pop_total, 1]) + np.random.rand(pop_total, D) * np.tile(VarMax, [pop_total, 1]) - \
              np.tile(VarMin, [pop_total, 1])
    ages = np.zeros((1, pop_total))
    packs = np.random.permutation(pop_total).reshape(n_packs, n_coy)
    coy_pack = np.tile(n_coy, [n_packs, 1])

    for c in range(pop_total):
        costs[0, c] = FOBJ(coyotes[c, :])

    nfeval = pop_total
    GlobalMin = min(costs)
    ibest = costs.argmin()
    GlobalParams = coyotes[ibest, :]

    year = 1
    while nfeval < nfevalMAX:
        year += 1
        for p in range(n_packs):
            coyotes_aux = coyotes[packs[p, :], :]
            costs_aux = costs[0, packs[p, :]]
            ages_aux = ages[0, packs[p, :]]
            n_coy_aux = coy_pack[p, 0]

            ind = np.argsort(costs_aux)
            costs_aux = costs_aux[ind]
            coyotes_aux = coyotes_aux[ind, :]
            ages_aux = ages_aux[ind]
            c_alpha = coyotes_aux[0, :]

            tendency = np.median(coyotes_aux, 0)
            new_coyotes = np.zeros((n_coy, D))

            for c in range(n_coy_aux):
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(n_coy_aux)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(n_coy_aux)

                new_coyotes[c, :] = coyotes_aux[c, :] + np.random.rand()*(c_alpha - coyotes_aux[rc1, :]) + \
                                    np.random.rand()*(tendency - coyotes_aux[rc2, :])

                for abc in range(D):
                    new_coyotes[c, abc] = max([min([new_coyotes[c, abc], VarMax[abc]]), VarMin[abc]])

                new_cost = FOBJ(new_coyotes[c, :])
                nfeval += 1

                if new_cost < costs_aux[c]:
                    costs_aux[c] = new_cost
                    coyotes_aux[c, :] = new_coyotes[c, :]

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
            p2[0, pdr[2:]] = r > 1-prob2

            n = np.logical_not(np.logical_or(p1, p2))

            pup = p1*coyotes_aux[parents[0], :] + \
                  p2*coyotes_aux[parents[1], :] + \
                  n*(VarMin + np.random.rand(1, D) * (VarMax - VarMin))

            pup_cost = FOBJ(pup)
            nfeval += 1

            worst = np.flatnonzero(costs_aux > pup_cost)

            if len(worst) > 0:
                older = np.argsort(ages_aux[worst])
                which = worst[older[::-1]]
                coyotes_aux[which[0], :] = pup
                costs_aux[which[0]] = pup_cost
                ages_aux[which[0]] = 0

            coyotes[packs[p], :] = coyotes_aux
            costs[0, packs[p]] = costs_aux
            ages[0, packs[p]] = ages_aux

        if n_packs > 1:
            if np.random.rand() < p_leave:
                rp = np.random.permutation(n_packs)[:2]
                rc = [np.random.permutation(coy_pack[rp[0], 0])[0], np.random.permutation(coy_pack[rp[1], 0])[0]]
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
    f = lambda x: np.sum(x**2)
    d = 30
    lu = np.zeros((2, d))
    lu[0, :] = -100
    lu[1, :] = 100
    nfeval = 1000*d
    n_packs = 10
    n_coy = 10
    t = time.time()
    y = np.zeros((1, 100))
    for i in range(100):
        mini, par = COA(f, lu, nfeval, n_packs, n_coy)
        y[0, i] = mini
        #print(time.time()-t)
        t = time.time()
    print([np.min(y), np.mean(y), np.median(y), np.max(y), np.std(y)])


