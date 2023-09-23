import numpy as np
from modify import *
import matplotlib.pyplot as plt


def simulate_hawkes(mu, alpha, beta, window):
    t = 0
    points = []
    while t < window:
        m = hawkes_intensity(mu, alpha, points, beta, t)
        s = np.random.exponential(scale = 1/m)
        ratio = hawkes_intensity(mu, alpha, points, beta, t+s)/m
        if ratio >= np.random.uniform():
            if t+s > window: break
            points.append(t+s)
        t = t+s
    return points

def hawkes_intensity(mu, alpha, points, beta, t):
    p = np.array(points)
    p = p[p <= t]
    p = alpha*beta*np.exp(beta*(p - t))
    return mu + np.sum(p)

def binned_process(data, window, binsize=1):
    '''
    Transform the truth origin data to binned data.
    
    Iutput: 
        data: the truth origin data (tpp)
        window: total length of time
        binsize: the width of bin

    Output:
        binned_data: the binned data (count data)
    '''
    data = np.array(data)
    lb = np.array([i for i in range(0,window,binsize)])
    ub = lb+binsize
    if ub[-1] > window: ub[-1] = window

    binned_data = []
    for l,u in zip(lb, ub):
        counts = np.sum([(data>l)&(data<=u)])
        binned_data.append(counts)
    return binned_data

def RelErr(gdth, est):
    '''
    gdth: groundtruth
    est: estimation
    '''
    gdth = np.array([gdth])
    est = np.array([est])

    dim = len(gdth)
    
    gdth = gdth.flatten()
    est  = est.flatten()

    if len(gdth) == dim and len(est) != dim :
        # the parameter should be mu and dim>1
        relerr = (np.sum(abs(gdth-est)/gdth))/dim
    else:
        # the parameter should be alpha or dim==1
        relerr = (np.sum(abs(gdth-est)/gdth))/np.square(dim)

    return relerr

def plot_events(origin_sample_event, modify_sample_event, true_event, interval):
    fig = plt.figure(figsize = (10,4))
    ax = plt.gca()

    plt.plot(true_event, [1]*len(true_event), 'bo', alpha = 0.3, markersize = 10)
    plt.plot(modify_sample_event, [0]*len(modify_sample_event), 'bo', alpha = 0.3, markersize = 10)
    plt.plot(origin_sample_event, [-1]*len(origin_sample_event), 'bo', alpha = 0.3, markersize = 10)

    ax.set_yticks([1, 0 ,-1])
    ax.set_yticklabels(['True', 'Modify', 'Origin'])
    
    ax.set_xlim(interval)
    ax.set_ylim([-1.5,1.5])
    ax.set_xlabel('Time')

########################### optimize method ############################
def optimize(sample_complete_data, lb, ub, target, method, mu, alpha, beta):
    if ub - lb < 0.1:
        return (ub-lb)/2
    else:
        accuracy = round((ub-lb)/0.1)
    dt = (ub-lb)/accuracy
    time_unit_sample = np.array([(lb+i*dt) for i in range(accuracy)])
    time_unit_sample = ((time_unit_sample+dt)+time_unit_sample)/2

    if method == 'order':
        intensity_unit_sample = [cumulative_intensity(sample_complete_data, time, mu, alpha, beta) for time in time_unit_sample]
    elif method == 'exact':
        intensity_unit_sample = [cdf_interarrival_prob(sample_complete_data, time, lb, ub,  mu, alpha, beta) for time in time_unit_sample]
    elif method == 'offspring':
        intensity_unit_sample = [cdf_offspring_distri(time, lb, ub, alpha, beta) for time in time_unit_sample]

    intensity_unit_sample = np.array(intensity_unit_sample)
    error_unit_sample = abs(intensity_unit_sample - target)
    min_error = np.min(error_unit_sample)
    index = np.where(error_unit_sample == min_error)[0][0]
    insert_time = time_unit_sample[index]
                 
    return insert_time


