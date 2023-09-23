import numpy as np
from scipy.optimize import newton
from sklearn.metrics.pairwise import pairwise_distances
from utils import *

#############################################################################
############################ Thinning Method ################################
#############################################################################
def conditional_arrival_probability(sample_complete_data, interval, mu, alpha, beta):
    '''
    Calculate the conditional arrival probability of each event in interval. Note: the sample complete data is
    necessary since the events happend before objective interval will influence the CAP of event in interval.
    Input:
        sample_complete_data: the complete simulated data (tpp).
        interval: the objective interval that will be thinned e.g. [0, 10]
    Output:
        CAP: a list that containes CAP of each event in interval.
    '''
    CAP = []
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>=lb)&(sample_complete_data<ub)]
    N = len(sample_complete_data)
    n = len(sample_interval_data)

    if n == 0: return 'There is no point in this interval.'

    # N*N mat
    diff_mat = pairwise_distances(sample_complete_data.reshape(N,1), metric = 'euclidean')
    kern_mat = 1 - np.exp(-beta*diff_mat)
    kern_mat[np.triu_indices(N)] = 0
    
    #-----------calc first point cond arrival prob------------#
    diff_lb = lb - sample_complete_data[sample_complete_data<lb] 
    kern_lb = 1 - np.exp(-beta*diff_lb)
    
    first_point = sample_interval_data[0]
    index_first_point = int(np.where(sample_complete_data == first_point)[0])

    if index_first_point == 0: # there's no points before lb
        cap_first_point = mu*(first_point)
    else:  # there must be points before lb
        term1_first = mu*(first_point - lb)
        term2_first = alpha*(np.sum(kern_mat, axis = 1)[index_first_point])
        term3_first = alpha*(np.sum(kern_lb))
        integral_first = term1_first + term2_first - term3_first
        cap_first_point = 1 - np.exp(-integral_first)

    CAP.append(cap_first_point)
    #----------------------------------------------------------#

    for num in range(n-1):
        index = index_first_point + num + 1
        term1 = mu*diff_mat[index, index-1]
        term2 = alpha*(np.sum(kern_mat, axis = 1)[index])
        term3 = alpha*(np.sum(kern_mat, axis = 1)[index-1])
        integral = term1+term2-term3
        cap = 1 - np.exp(-integral)
        CAP.append(cap) 

    return CAP

def numerical_test_cap(sample_complete_data, interval, unit_num, mu, alpha, beta):
    '''
    Numerical method to calculate CAP.The output is same with conditional_arrival_probability.

    unit_num: resolution i.e. the number of parts that the interval is divided.
    '''
    AREA = []
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>=lb)&(sample_complete_data<ub)]
    N = len(sample_complete_data)
    n = len(sample_interval_data)

    if n == 0: return 'There is no point in this interval.'
    
    # list unit_sample according to interval and unit_num
    dt = (ub-lb)/unit_num
    unit_sample = [lb+i*dt for i in range(unit_num)]
    unit_sample = np.array(unit_sample)

    # calc the _hawkesintensity at unit_sample
    unit_intensity = []
    for point in unit_sample:
        intensity = hawkes_intensity(mu, alpha, sample_complete_data, beta, point)
        unit_intensity.append(intensity)
    unit_intensity = np.array(unit_intensity)
    
    # calc area of each point happened in interval
    for num in range(n):
        if num == 0: lb_point = lb
        else: lb_point = sample_interval_data[num-1]
        ub_point = sample_interval_data[num]

        poi_uni = unit_sample[(unit_sample>=lb_point)&(unit_sample<ub_point)]
        ind_uni = [int(np.where(unit_sample == i)[0]) for i in poi_uni]
        int_uni = unit_intensity[ind_uni]
        area = np.sum(int_uni*dt)
        AREA.append(area)

    AREA = 1 - np.exp(-np.array(AREA))
    return AREA

def thinning(sample_complete_data, true_interval_counts, interval, mu, alpha, beta):
    '''
    Default that sample_complete_data should more than true_complete_data
    Output: thinned sample interval data 
    '''
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>=lb)&(sample_complete_data<ub)]
    
    thinning_num = len(sample_interval_data) - true_interval_counts
    if thinning_num <= 0: return 'Error: no need to thinning'
    if true_interval_counts == 0: return []

    CAP = conditional_arrival_probability(sample_complete_data, interval, mu, alpha, beta)

    thinning_CAP = np.sort(CAP)[:thinning_num]
    thinning_ind = [int(np.where(CAP == i)[0]) for i in thinning_CAP]
    DISCARD_POI = sample_interval_data[thinning_ind]
    mask = np.isin(sample_interval_data, DISCARD_POI)
    PRESERVED_POI  = sample_interval_data[~mask]

    if true_interval_counts == len(PRESERVED_POI): pass
    else: print('Error: the length diff with truth after thinned.')
    return PRESERVED_POI

#############################################################################
######################### Thickening Order Method ###########################
#############################################################################
def cumulative_intensity(sample_complete_data, time, mu, alpha, beta):
    '''
    Compute cumulative intensity at time t given complete data and parameters.
    Input:
        sample_complete_data: 
        time: the time that needed to be calculated cumulative intensity at.
    
    Output:
        cumulative intensity at time t.
    '''
    sample_complete_data = np.array(sample_complete_data)
    berfor_event_time = sample_complete_data[sample_complete_data < time]
    if len(berfor_event_time) == 0: return mu*time
    diff = time - berfor_event_time
    return mu*time + alpha*(np.sum(1 - np.exp(-beta*diff)))
    

def inverse_cumulative_intensity(sample_complete_data, target_intensity, interval, mu, alpha, beta):
    lb = interval[0]
    ub = interval[1]

    lb_cit = cumulative_intensity(sample_complete_data, lb, mu, alpha, beta)
    ub_cit = cumulative_intensity(sample_complete_data, ub, mu, alpha, beta)

    if target_intensity<lb_cit or target_intensity>ub_cit: return 'Target intensity beyong the interval bound.'
    
    # Define a function that calculates the difference between the target intensity and the cumulative intensity function
    def equation(time):
        return cumulative_intensity(sample_complete_data, time, mu, alpha, beta) - target_intensity

    try:
        # Use a root-finding algorithm (Newton-Raphson method) to solve for the time
        initial_guess = interval[0]
        time_solution = newton(equation, initial_guess)
    except:
        print('Newton Raphson failed')
        time_solution = np.random.uniform(lb, ub)
    
    #intensity_error = cumulative_intensity(sample_complete_data, time_solution, mu, alpha, beta) - target_intensity
    #print('Intensity Error:', intensity_error)
    
    return time_solution


def thickening_order(sample_complete_data, true_interval_counts, interval, mu, alpha, beta):
    '''
    Default that sample_complete_data should more than true_complete_data
    Output: thickenned sample interval data 
    '''
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>=lb)&(sample_complete_data<ub)]
    
    thickening_num = true_interval_counts - len(sample_interval_data)
    if thickening_num <= 0: return 'Error: no need to thickening'

    # calc the equal points of intensity
    lb_cit = cumulative_intensity(sample_complete_data, lb, mu, alpha, beta)
    ub_cit = cumulative_intensity(sample_complete_data, ub, mu, alpha, beta)
    insert_intensity = [lb_cit+((ub_cit-lb_cit)/(thickening_num+1))*i for i in range(1,thickening_num+1)]

    INSERT_POI = []
    for int in insert_intensity:
        insert_time = inverse_cumulative_intensity(sample_complete_data, int, interval, mu, alpha, beta)
        INSERT_POI.append(insert_time)

    THICKENING_POI = np.append(sample_interval_data, INSERT_POI)
    THICKENING_POI = np.sort(THICKENING_POI)
    if true_interval_counts == len(THICKENING_POI): pass
    else: print('Error: the length diff with truth after thickened.')
    
    return THICKENING_POI
#############################################################################
################### Thickening Expectation Method ###########################
#############################################################################
def numerical_to_calc_expection_thickening_insert_time(sample_complete_data, ti_1, ti, unit_num, mu, alpha, beta):
    # Note: the accuracy of numerical method may result in error of time beyong bound
    dt = (ti - ti_1)/unit_num
    unit_sample = [ti_1+i*dt for i in range(unit_num)]
    unit_sample = np.array(unit_sample)

    # numerator
    unit_area = []
    term3 = cumulative_intensity(sample_complete_data, ti_1, mu, alpha, beta)
    for t in unit_sample:
        term1 = hawkes_intensity(mu, alpha, sample_complete_data, beta, t)
        term2 = cumulative_intensity(sample_complete_data, t, mu, alpha, beta)
        objective_fun = t*term1*np.exp(-(term2 - term3))
        unit_area.append(objective_fun*dt)
    numerator = np.sum(unit_area)

    # denominator
    term4 = cumulative_intensity(sample_complete_data, ti, mu, alpha, beta)
    term5 = cumulative_intensity(sample_complete_data, ti_1, mu, alpha, beta)
    denominator = 1 - np.exp(-(term4 - term5))

    # result
    insert_time = numerator/denominator

    # check
    # sometime may beyong the time bound
    if insert_time < ti_1 or insert_time > ti:
        if abs(insert_time - ti_1) < 0.1:
            insert_time = (ti + ti_1)/2
        elif abs(insert_time - ti) < 0.1:
            insert_time = (ti + ti_1)/2
        else: 
            #print(ti_1, ti, insert_time)
            #raise ValueError('Error: Thickening time beyond the bound.')
            insert_time = (ti + ti_1)/2
    return insert_time

def find_max_arrival_prob_interarrival_interval(sample_complete_data, sample_interval_data, lb, ub, mu, alpha, beta):
    # need to calc len(sample_interval_data)+1 arrival_probability
    arrival_probability = []
    for index in range(len(sample_interval_data)+1):
        if index == 0:
            bigger_cit = cumulative_intensity(sample_complete_data, sample_interval_data[0], mu, alpha, beta)
            smaller_cit = cumulative_intensity(sample_complete_data, lb, mu, alpha, beta)
        elif index == len(sample_interval_data):
            bigger_cit = cumulative_intensity(sample_complete_data, ub, mu, alpha, beta)
            smaller_cit = cumulative_intensity(sample_complete_data, sample_interval_data[-1], mu, alpha, beta)
        else:
            bigger_cit = cumulative_intensity(sample_complete_data, sample_interval_data[index], mu, alpha, beta)
            smaller_cit = cumulative_intensity(sample_complete_data, sample_interval_data[index-1], mu, alpha, beta)
        ap = bigger_cit - smaller_cit
        arrival_probability.append(ap)
    
    # find the maximum arrival probability
    max_ap = np.max(arrival_probability)
    max_ap_index = int(np.where(arrival_probability == max_ap)[0])
    if max_ap_index == len(sample_interval_data):
        ti = ub
        ti_1 = sample_interval_data[-1]
    elif max_ap_index == 0:
        ti = sample_interval_data[0]
        ti_1 = lb   
    else:
        ti = sample_interval_data[max_ap_index]
        ti_1 = sample_interval_data[max_ap_index - 1]
    return ti_1, ti

def thickening_expectation(sample_complete_data, true_interval_counts, interval, mu, alpha, beta):
    unit_num = 1000
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>lb)&(sample_complete_data<=ub)]
    
    thickening_num = true_interval_counts - len(sample_interval_data)
    if thickening_num <= 0: return 'Error: no need to thickening'
    
    if len(sample_interval_data) == 0:
        insert_time = np.random.uniform(lb, ub, thickening_num)
        sample_interval_data = np.sort(insert_time)
    else:
        # calc each area between any neighbour points or bound
        for _ in range(thickening_num):
            ti_1, ti = find_max_arrival_prob_interarrival_interval(sample_complete_data, sample_interval_data, lb, ub, mu, alpha, beta)
            insert_time = numerical_to_calc_expection_thickening_insert_time(sample_complete_data, ti_1, ti, unit_num, mu, alpha, beta)
            #if insert_time <lb or insert_time >ub: print('Error: insert time beyond bound.')
            
            # update complete data and interval data
            sample_complete_data = np.append(sample_complete_data, insert_time)
            sample_complete_data = np.sort(sample_complete_data)
            sample_interval_data = np.append(sample_interval_data, insert_time)
            sample_interval_data = np.sort(sample_interval_data)

    # check

    if true_interval_counts == len(sample_interval_data): pass
    else: print('Error: the length diff with truth after thickened.')
    
    return sample_interval_data
#############################################################################
######################### Thickening Exact Method ###########################
#############################################################################
def cdf_interarrival_prob(sample_complete_data, t, ti_1, ti, mu, alpha, beta):
    '''
    if t == ti_1: the result cdf should be 0
    if t == ti: the result cdf should be 1
    '''
    # Newton raphson may cause time beyong the bound.
    #if t<ti_1 or t>ti: raise ValueError('Time beyong the interarrival bound.')
    
    term1 = cumulative_intensity(sample_complete_data, ti_1, mu, alpha, beta)
    term2 = cumulative_intensity(sample_complete_data, ti, mu, alpha, beta)
    term3_variable = cumulative_intensity(sample_complete_data, t, mu, alpha, beta)
    cdf = (1-np.exp(-(term3_variable - term1)))/(1-np.exp(-(term2 - term1)))
    return cdf


def inverse_cdf_interarrival_prob(sample_complete_data, target_cdf, ti_1, ti, mu, alpha, beta):
    '''
    target_cdf: should belong to range (0,1)
    '''
    def equation(time):
        return cdf_interarrival_prob(sample_complete_data, time, ti_1, ti, mu, alpha, beta) - target_cdf

    try:
        initial_guess = ti_1
        time_solution = newton(equation, initial_guess)
    except:
        print("Newton Raphson failed")
        time_solution = np.random.uniform(ti_1, ti)

    #intensity_error = cdf_interarrival_prob(sample_complete_data, time_solution, ti_1, ti, mu, alpha, beta) - target_cdf
    #print('Intensity Error:', intensity_error)

    return time_solution

def thickening_exact(sample_complete_data, true_interval_counts, interval, mu, alpha, beta):
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>lb)&(sample_complete_data<=ub)]
    
    thickening_num = true_interval_counts - len(sample_interval_data)
    if thickening_num <= 0: return 'Error: no need to thickening'
    

    if len(sample_interval_data) == 0:
        insert_time = np.random.uniform(lb, ub, thickening_num)
        sample_interval_data = np.sort(insert_time)
    else:
        # calc each area between any neighbour points or bound
        for _ in range(thickening_num):
            ti_1, ti = find_max_arrival_prob_interarrival_interval(sample_complete_data, sample_interval_data, lb, ub, mu, alpha, beta)
            target_cdf = np.random.uniform(0,1)
            insert_time = inverse_cdf_interarrival_prob(sample_complete_data, target_cdf, ti_1, ti, mu, alpha, beta)
        
            # update complete data and interval data
            sample_complete_data = np.append(sample_complete_data, insert_time)
            sample_complete_data = np.sort(sample_complete_data)
            sample_interval_data = np.append(sample_interval_data, insert_time)
            sample_interval_data = np.sort(sample_interval_data)

    # check
    if true_interval_counts == len(sample_interval_data): pass
    else: print('Error: the length diff with truth after thickened.')

    return sample_interval_data
#############################################################################
##################### Thickening Offspring Method ###########################
#############################################################################
def cdf_offspring_distri(t, ti_1, ti, alpha, beta):
    '''
    Don't depend on sample history.
    if t == ti_1: the result cdf should be 0
    if t == ti: the result cdf should be 1

    calc the conditional prob that will be inversed to calc time.
    '''
    term1 = alpha*(1-np.exp(-beta*ti_1))
    term2 = alpha*(1-np.exp(-beta*ti))
    term3_variable = alpha*(1-np.exp(-beta*t))
    if abs(term1 - term2) < 0.000001:
        # sometimes ti_1 and ti is close enough to let term1 = term2
        # we can return a random that means there's no difference to choice any unit sample
        return np.random.uniform(0,1) 
    ofs = (1-np.exp(-(term3_variable-term1)))/(1-np.exp(-(term2-term1)))
    return ofs

def inverse_cdf_offspring_distri(target_cdf, ti_1, ti, alpha, beta):
    '''
    Get one insert time.
    '''
    time_solution = optimize(None, ti_1, ti, target_cdf, 'offspring', None, alpha, beta)
    return time_solution

def thickening_offspring(sample_complete_data, true_interval_counts, interval, mu, alpha, beta):
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>lb)&(sample_complete_data<=ub)]
    
    thickening_num = true_interval_counts - len(sample_interval_data)
    if thickening_num <= 0: return 'Error: no need to thickening'
    
    if len(sample_interval_data) == 0:
        insert_time = np.random.uniform(lb, ub, thickening_num)
        sample_interval_data = np.sort(insert_time)
    else:
        # calc each area between any neighbour points or bound
        for _ in range(thickening_num):
            ti_1, ti = find_max_arrival_prob_interarrival_interval(sample_complete_data, sample_interval_data, lb, ub, mu, alpha, beta)
            target_cdf = np.random.uniform(0,1)
            insert_time = inverse_cdf_offspring_distri(target_cdf, ti_1, ti, alpha, beta)
        
            # update complete data and interval data
            sample_complete_data = np.append(sample_complete_data, insert_time)
            sample_complete_data = np.sort(sample_complete_data)
            sample_interval_data = np.append(sample_interval_data, insert_time)
            sample_interval_data = np.sort(sample_interval_data)

    # check
    if true_interval_counts == len(sample_interval_data): pass
    else: print('Error: the length diff with truth after thickened.')

    return sample_interval_data


########################### random method ############################
def random_thinning(sample_complete_data, true_interval_counts, interval):
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>=lb)&(sample_complete_data<ub)]
    
    thinning_num = len(sample_interval_data) - true_interval_counts
    if thinning_num <= 0: return 'Error: no need to thinning'
    if true_interval_counts == 0: return []

    sample_interval_data = np.random.choice(sample_interval_data, true_interval_counts, replace = False)
    return sample_interval_data

def random_thickening(sample_complete_data, true_interval_counts, interval):
    lb = interval[0]
    ub = interval[1]
    sample_complete_data = np.array(sample_complete_data)
    sample_interval_data = sample_complete_data[(sample_complete_data>=lb)&(sample_complete_data<ub)]
    
    thickening_num = true_interval_counts - len(sample_interval_data)
    if thickening_num <= 0: return 'Error: no need to thickening'

    insert_time = np.random.uniform(lb, ub, thickening_num)
    sample_interval_data = np.append(sample_interval_data, insert_time)
    return sample_interval_data
    