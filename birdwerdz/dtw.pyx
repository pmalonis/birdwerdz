#!/usr/bin/env python
"""
Functions used to apply birdsong recognition algorithm as described in
Fantana and Kozhevnikov 2014
"""

import numpy as _np
cimport numpy as _np
cimport cython
import scipy.signal as _signal
import libtfr as _ltfr
import scipy.cluster.vq as _vq
from scipy.interpolate import interp1d as _interp1d
_DTYPE = _np.float64
ctypedef _np.float64_t _DTYPE_t

cdef inline double d_min(double a, double b): return a if a <= b else b
cdef extern void distance(_DTYPE_t* a, _DTYPE_t* b, _DTYPE_t* dist_mat_ptr, int length) 
cdef extern void assign_D (_DTYPE_t* distance, _DTYPE_t* D_entry, int* T_entry, int nargs, ... )     

   
def process_recording(signal, fs, 
                      win_len=256, noise_cutoff=500, 
                      numtaps=101, t_step=.001):

    """
    Applies a high-pass filter to the data returns the spectrogram   
    
    Parameters
    ----------
    signal - signal to be analyzed
    fs - sampling rate
    win_len - length of window used to compute spectrogram, in samples
    noise_cutoff - cutoff frequency for high pass filter
    numtaps - Number of filter taps 
    t_step - Time step of spectrogram in seconds
    
    """
    #filter parameters
    noise_cutoff= 500
    numtaps = 101

    window = _signal.hann(win_len)

    b = _signal.firwin(numtaps, noise_cutoff, pass_zero=False,  nyq=fs/2.)
    filtered = _signal.lfilter(b, 1, signal)    
    # ensuring that log spectrogram doesn't give -inf
    spec=_np.abs(_ltfr.stft(filtered, window, int(tstep*fs)))
    spec[spec==0]=_np.min(spec[spec!=0])
    log_spec = _np.log(spec)

    max_freq = 10000
    max_idx = int(max_freq/(float(fs)/win_len))

    return log_spec[:max_idx,:]


@cython.boundscheck(False)
def _dist_mat(_np.ndarray[_DTYPE_t, ndim=2] V_spec, 
              _np.ndarray[_DTYPE_t, ndim=2] T_spec):
    """
    Creates the distance matrix between the vocalization spectrogram
    """
    V_spec = _np.asfortranarray(V_spec)
    T_spec = _np.asfortranarray(T_spec)

    cdef int i
    cdef int j
    cdef int i_max = V_spec.shape[1]
    cdef int j_max = T_spec.shape[1]
    cdef _np.ndarray[_DTYPE_t, ndim=2] d = _np.zeros([i_max, j_max], dtype=_DTYPE)
    spec_len = T_spec.shape[0]

    for i in xrange(i_max):
        for j in xrange(j_max):
            distance(&V_spec[0,i], &T_spec[0,j], &d[i,j], spec_len)

    return d

@cython.boundscheck(False)
def _accumulated_dist(_np.ndarray[_DTYPE_t, ndim=2] d):
    """
    Calculates minimum accumulated distance among all paths to 
    each grid point of the distance matrix.

    Parameters:
    d_matrix - Distance matrix

    Returns:

    D - accumulated distance matrix 
    T - saved path decisions for backtracking
    """
    
    cdef int i_max = d.shape[0]
    cdef int j_max = d.shape[1]
    cdef _np.ndarray[_DTYPE_t, ndim=2, mode = 'c'] D = _np.zeros([i_max,j_max], dtype=_DTYPE)
    cdef _np.ndarray[int, ndim=2, mode = 'c'] T = _np.zeros([i_max,j_max], dtype=_np.intc)

    cdef int i, j
    #initialize edges 
    for i in xrange(i_max):
        D[i,1] = d[i,0]

    for j in xrange(j_max):
        D[1,j] = d[0,j]

    for i in xrange(1, i_max):        
        D[i,2] = d[1,j] + d_min(D[i,1], D[i-1,1])

    for j in xrange(1, j_max):
        D[2,j] = d[i,1] + d_min(D[1,j], D[1,j-1])
                         
    cdef int nargs = 3
    for i in xrange(2, i_max):        
        for j in xrange(2, D.shape[1]):
            assign_D(&d[i,j], &D[i,j], &T[i,j], nargs,
                      D[i-2, j-2],
                      D[i-1, j-2],
                      D[i-2, j-1])

    return D,T


def _get_endpoints(D):
    """
    Finds endpoints of paths from accumulated distance matrix

    Parameters:
    
    D - accumulated distance matrix
    

    Returns:
    
    i_end - ends of paths, i.e. local minima of last column of 
            accumulated distance matrix, sorted in descending order


    """
    last_col = D[(D.shape[1]/2):, -1]   
    local_min, = _np.where((last_col[1:-1] < last_col[2:]) &
                               (last_col[1:-1] < last_col[:-2]))

    local_min += 1

    if last_col[0] < last_col[1]:
        local_min = _np.append(local_min, 0)
    
    if last_col[-1] < last_col[-2]:
        local_min = _np.append(local_min, len(last_col) - 1)    
    
    local_min = _np.array([lmin for lmin in _np.argsort(last_col) if lmin in local_min])

    #convert to index of last_col to index of D
    endpoints = local_min + (D.shape[1]/2)

    return endpoints
    

def _backtrack(int i_end, _np.ndarray[int,ndim=2] T):

    """
    Backtracks from endpoint of paths to find the full DTW path

    Parameters:
    
    i_end - endpoint of path
    T - saved path decisions for backtracking (second return argument 
    of "accumulated_dist")


    Returns:
    
    path - a matrix whose rows contain the index pair of each point in 
    the backtracked path, starting from the beginning of the path

    """

    cdef int i = i_end
    cdef int j = T.shape[1] - 1
    path = _np.zeros((T.shape[1], 2), dtype=int)
    path[0,:] = [i, j]
    cdef int idx = 1
    while i>0 and j>0:        
        if T[i,j] == 3:
            i -= 2
            j -= 1
        elif T[i,j] == 2:
            i -= 1
            j -= 2
        elif T[i,j] == 1:
            i -= 2
            j -= 2
        elif T[i,j] == 0:
            i -= 1
            j -= 1
        else:
            #stops path if intersects with other path (if T has been 
            #altered after finding the other path)
            return path

        path[idx,:] = [i,j]
        idx+=1

    return _np.flipud(path[:idx,:])



def _interpolate_path(path, d):

    """
    Parameters:
 
    path - indices of path as returned by "backtrack"
    d - distnace matrix (returned by "dist_mat")

    Returns:
    interp_path - array of indices of vocalization spectrogram
    (since interpolation ensures template  step size of 1)
    
    """


    interp_path = _np.array([path[0,0]],dtype=int)

    for idx, (i,j) in enumerate(path):
        if idx == 0: continue 
    
        i_step = i - path[idx-1,0]
        j_step = j - path[idx-1,1]

        # import pdb
        # pdb.set_trace()


        if i_step not in (1,2) or j_step not in (1,2):
            print path
            raise ValueError("improper step sizes in path")
        if j_step == 2:
            if i_step == 1:
                i_new =  i - _np.argmin([d[i, j-1], d[i-1, j-1]])
            if i_step == 2:
                i_new = i-1
            interp_path = _np.append(interp_path, i_new) 
        
        interp_path = _np.append(interp_path, i)

    return interp_path
                                

@cython.profile(True)
def _get_paths(D, T, d,  t_0=0):
      
    endpoints = _get_endpoints(D)    
    paths = _np.zeros((endpoints.shape[0], D.shape[1]), dtype=int)
    counter=0
    for i_end in endpoints:
   
        P= _backtrack(i_end, T)
        
        if P[0,1] != 0:
            continue
        else:
            paths[counter,:] = t_0 + _interpolate_path(P, d)       
            counter+=1
            #alter T to prevent path overlap 
            i_start = P[0]
            mid_path_bool = (P > (i_start + 0.1*(i_end-i_start))
                             ) & (P < (i_start + 0.9*(i_end-i_start)))
            i = _np.extract(mid_path_bool, P)
            
            T[i,:] = -1
            
    paths=_np.delete(paths,_np.s_[counter:],0)
    distances = D[paths[:,-1],-1]
    return paths, distances


def find_matches(vocalization, template, fs_voc, fs_temp, 
                 win_len=256, t_step=.001):
      
    """
    Finds potential matches to a template motif in a recorded vocalization.
    The matches are taken from the local minima last column of the minimum 
    accumulated distance matrix, and are ranked from the smallest distance
    to the template to the largest.  

    Parameters
    ----------
    
    vocalization - array containing the sampled data of the recording to be analyzed
    template - array containing the sampled data of the template motif
    fs_voc - sampling rate of the vocalization
    fs_temp - sampling rate of the template

    Returns three lists, each containing one entry for each match
    
    motif_intervals - a 2d array in which each row contains the start and stop sample
                      for each match

    spectrograms - a 3d array containing the temporally warped spectrograms of each match.
                   The first dimension represents individual matches, the second dimension
                   the frequency axis of the spectrogram, and the third the time axis.
                    
    dtw_paths - a 2d array in which each row contains the indices 
                of the columns of the vocalization spectrogram used in the temporally
                warped spectrograms
 
    distances - A 1d array containing the distance of each match's spectrogram to the 
                template spectrogram
    """
        
    T_spec = process_recording(template, fs_temp)
    V_spec = process_recording(vocalization, fs_voc)
    d=_dist_mat(V_spec, T_spec)
    D,T=_accumulated_dist(d)

    dtw_paths,distances=_get_paths(D, T, d)

    spectrograms = _np.zeros((dtw_paths.shape[0],) + T_spec.shape)
    for i,p in enumerate(dtw_paths):
        spectrograms[i,:,:] = V_spec[:,p]
    
    spec_step = int(tstep*fs_voc)
    motif_intervals = _np.array([_np.array([p[0], p[-1]])*spec_step+fft_res
                                 for p in dtw_paths])

    return motif_intervals, spectrograms, dtw_paths, distances
       

def cluster_motifs(spectrograms, nclusters=10):
    '''
    Cluster list of putative motif specrograms using k-means clustering
    '''
    amp_vectors = spectrograms.sum(1)
    centroids,_ = _vq.kmeans(amp_vectors, nclusters)
    id,_ = _vq.vq(amp_vectors, centroids)
    spec_shape = spectrograms.shape[1:]
    mean_spectrograms = _np.zeros((nclusters,) + spec_shape)
    for i in xrange(nclusters):
        mean_spectrograms[i,:,:] = spectrograms[id==i,:,:].mean(0)

    return id, mean_spectrograms

def align_events(dtw_path, events, fs_temp, fs_voc,
                 win_len=256, t_step=.001):
    '''
    Aligns events according to a dtw path
    Parameters
    ----------
    dtw_path - A 1d array containing the dtw path that will be used to align the 
               spikes
    events - A 1d array containing the event times to be aligned to the warpred spectrogram.
             The the times should be given in units of seconds. 
    fs_voc - sampling rate of the vocalization
    fs_temp - sampling rate of the template

    Returns
    -------
    aligned_events - 1d array containing the aligned events.  Events occuring outside the 
                     interval containing the template match will not be included
    '''

    fft_res = win_len/2

    spec_step_temp = int(tstep*fs_temp)
    spec_step_voc = int(tstep*fs_voc)
    
    #spectrogram time axis of template in samples
    temp_samples = _np.arange(len(dtw_path))*spec_step_temp + fft_res 

    #spectrogram time axis of vocalization in samples
    voc_samples = dtw_path*spec_step_voc + fft_res 
    
    interp = _interp1d(voc_samples, temp_samples) 
    filtered_events = _np.array([e for e in events if voc_samples[0] < e < voc_samples[-1]])
    aligned_events = interp(filtered_events)

    return aligned_events
