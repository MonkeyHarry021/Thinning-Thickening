import numpy as np
from modify import *
from sklearn.metrics.pairwise import pairwise_distances

def correction(sample_seq, true_counts, window, binsize,
               mhat, ahat, bhat,
               thick_method = 'order',
               thin_method = 'thinning',
               verbose = True):
    '''
    Input:
        sample_seq: simulated point process
        mhat, ahat, bhat: the estimation of EM
    Output:
        correction_seq: corrected point process
    '''
    correction_seq = []
    sample_seq = np.array(sample_seq)
    sample_counts = binned_process(sample_seq, window, binsize = binsize)
    lb = np.array([i for i in range(0, window, binsize)])
    ub = lb + binsize
    if ub[-1] > window: ub[-1] = window

    # loop for each interval
    interval = np.concatenate((lb.reshape(len(lb), 1), ub.reshape(len(ub), 1)), axis=1)
    for index in range(len(interval)):

        inter = interval[index]
        if sample_counts[index] > true_counts[index]:
            # thinning
            if thin_method == 'thinning':
                thinning_seq = thinning(sample_seq, true_counts[index], inter, mhat, ahat, bhat)
            elif thin_method == 'random':
                thinning_seq = random_thinning(sample_seq, true_counts[index], inter)
            else: raise TypeError('No suck thinning method.')
            correction_seq = np.append(correction_seq, thinning_seq)
        
        elif sample_counts[index] < true_counts[index]:
            # thickening
            if thick_method == 'order':
                thickening_seq = thickening_order(sample_seq, true_counts[index], inter, mhat, ahat, bhat)
            elif thick_method == 'expectation':
                thickening_seq = thickening_expectation(sample_seq, true_counts[index], inter, mhat, ahat, bhat)
            elif thick_method == 'exact':
                thickening_seq = thickening_exact(sample_seq, true_counts[index], inter, mhat, ahat, bhat)
            elif thick_method == 'offspring':
                thickening_seq = thickening_offspring(sample_seq, true_counts[index], inter, mhat, ahat, bhat)
            elif thick_method == 'random':
                thickening_seq = random_thickening(sample_seq, true_counts[index], inter)
            else: raise TypeError('No shuch thickening method.')
            correction_seq = np.append(correction_seq, thickening_seq)
        
        elif sample_counts[index] == true_counts[index]:
            sample_interval_data = sample_seq[(sample_seq>inter[0])&(sample_seq<=inter[1])]
            correction_seq = np.append(correction_seq, sample_interval_data)
    correction_seq = np.sort(correction_seq)
    # check
    correction_counts = binned_process(correction_seq, window, binsize = binsize)
    if len(correction_counts) == len(true_counts):
        compare_bool = (np.array(correction_counts) == np.array(true_counts))
        if False not in compare_bool:
            if verbose == True: print('Accomplish Correction')
        else:
            print('Error: some intervals are not modified correctly.')
    else:
        print('Error: the bin num is different between truth and correction.')

    return correction_seq

def Correct_EM_univar(true_counts, old_mhat, old_ahat, old_what, 
                      window, binsize, maxiter, epsilon = 0.001, 
                      thick_method = 'order',
                      thin_method = 'thinning',
                      verbose = True):
    '''
    mhat: initial value for estimation of mu
    ahat: initial value for estimation of alpha
    what: initial value for estimation of omega
    maxiter: maximum iteration count
    window: the maximum duration that hawkes process would happen
    '''
    k = 0
    N  = np.sum(true_counts)
    Tm = window
    log_ll = []

    while k <= maxiter:

        sample_points = simulate_hawkes(old_mhat, old_ahat, old_what, Tm)
        points = correction(sample_points, true_counts, Tm, binsize,
                            old_mhat, old_ahat, old_what,
                            thick_method, thin_method, verbose)
        points = np.array(points)
        points = points[points<Tm]

        diffs = pairwise_distances(np.array([points]).T, metric = 'euclidean')
        diffs[np.triu_indices(N)] = 0
        kern = old_what*np.exp(-old_what*diffs)

        ag = np.multiply(old_ahat, kern)
        ag[np.triu_indices(N)] = 0  # N*N

        rates = old_mhat + np.sum(ag, axis = 1) # N*1

        # the diagonal entries is 0, because of ag
        p_ii = np.divide(old_mhat, rates) # N*1
        p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1, N))) # N*N

        mhat = np.sum(p_ii)/Tm
        ahat = np.sum(p_ij.flatten())/N
        #what = np.sum(p_ij.flatten())/np.sum((np.multiply(diffs, p_ij)).flatten())
        what = 1 # fix beta

        #if k % 5 == 0:
        term1 = np.sum(np.log([hawkes_intensity(mhat, ahat, points,what,t) for t in points]))
        term2 = mhat*Tm
        term3 = ahat*np.sum(1 - np.exp(-what*(Tm-points)))
        new_LL = -(term1 - term2 - term3)
        log_ll.append(new_LL)
        if verbose == True:
            print('After ITER %d (Negative log likelihood: %1.3f)'%(k, new_LL))
        #if abs(new_LL - old_LL) <= epsilon:
        old_parameter = np.array([old_mhat, old_ahat, old_what])
        new_parameter = np.array([mhat, ahat, what])

        if np.linalg.norm(new_parameter - old_parameter) < epsilon:
            if verbose == True:
                print('Reached stopping criterion')
                print('mhat is:', mhat)
                print('ahat is:', ahat)
                print('what is:', what)
            break
        
        old_mhat = mhat
        old_what = what
        old_ahat = ahat
        k += 1

    if verbose == True:
        if k == maxiter+1:
            print('Reached max iter %d.'% (maxiter))
            print('ahat is:', ahat)
            print('mhat is:', mhat)
            print('what is:', what)

    return mhat, ahat, what, log_ll


def EM_univar(points, old_mhat, old_ahat, old_what, window, maxiter, epsilon = 0.01):
    '''
    mhat: initial value for estimation of mu
    ahat: initial value for estimation of alpha
    what: initial value for estimation of omega
    maxiter: maximum iteration count
    window: the maximum duration that hawkes process would happen
    '''
    k = 0
    Tm = window
    N  = len(points)

    points = np.array(points)
    points = points[points<Tm]
    diffs = pairwise_distances(np.array([points]).T, metric = 'euclidean')
    diffs[np.triu_indices(N)] = 0
  
    log_ll = []
    term1 = np.sum(np.log([hawkes_intensity(old_mhat, old_ahat, points,old_what,t) for t in points]))
    term2 = old_mhat*Tm
    term3 = old_ahat*np.sum(1 - np.exp(-old_what*(Tm-points)))
    old_LL = -(term1 - term2 - term3)
    log_ll.append(old_LL)

    while k <= maxiter:

        kern = old_what*np.exp(-old_what*diffs)
        ag = np.multiply(old_ahat, kern)
        ag[np.triu_indices(N)] = 0  # N*N

        rates = old_mhat + np.sum(ag, axis = 1) # N*1

        # the diagonal entries is 0, because of ag
        p_ii = np.divide(old_mhat, rates) # N*1
        p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1, N))) # N*N

        mhat = np.sum(p_ii)/Tm
        ahat = np.sum(p_ij.flatten())/N
        what = np.sum(p_ij.flatten())/np.sum((np.multiply(diffs, p_ij)).flatten())

        #k % 5 == 0:
        #term1 = np.sum(np.log(rates))
        #term2 = mhat*Tm + ahat*np.sum(1 - np.exp(what*(np.array(points) - Tm)))
        term1 = np.sum(np.log([hawkes_intensity(mhat, ahat, points,what,t) for t in points]))
        term2 = mhat*Tm
        term3 = ahat*np.sum(1 - np.exp(-what*(Tm-points)))

        new_LL = term1 - term2 - term3
        log_ll.append(new_LL)
        print('After ITER %d (old: %1.3f new: %1.3f)'%(k, old_LL, new_LL))
        #if abs(new_LL - old_LL) <= epsilon:
        old_parameter = np.array([old_mhat, old_ahat, old_what])
        new_parameter = np.array([mhat, ahat, what])
        if np.linalg.norm(new_parameter - old_parameter) < epsilon:
            print('Reached stopping criterion')
            print('mhat is:', mhat)
            print('ahat is:', ahat)
            print('what is:', what)
            break
        old_LL = new_LL
        old_mhat = mhat
        old_what = what
        old_ahat = ahat
        k += 1

    if k == maxiter+1:
        print('Reached max iter %d.'% (maxiter))
        print('mhat is:', mhat)
        print('ahat is:', ahat)
        print('what is:', what)

    return mhat, ahat, what, log_ll
