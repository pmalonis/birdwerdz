import numpy as np
cimport numpy as np
cimport cython
from scipy.signal import firwin, lfilter, hann
import h5py
from itertools import izip
from scipy.fftpack import fft
import libtfr


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline double d_min(double a, double b): return a if a <= b else b
cdef extern void distance(DTYPE_t* a, DTYPE_t* b, DTYPE_t* dist_mat_ptr, int length) 
cdef extern void assign_D (DTYPE_t* distance, DTYPE_t* D_entry, int* T_entry, int nargs, ... )                     


#spectrogram parameters
tstep = .001
fft_res = 128
win_len = fft_res*2
window = hann(win_len)
   
def process_recording(signal, fs):
    """
    Applies a high-pass filter to the data and computes the spectrogram
    
    """
    #filter parameters
    noise_cutoff= 500
    numtaps = 101


    b = firwin(numtaps, noise_cutoff, pass_zero=False,  nyq=fs/2.)
    filtered = lfilter(b, 1, signal)    
    spec=libtfr.stft(filtered, window, int(tstep*fs))

    log_amp = np.log(np.abs(spec))

    return log_amp


@cython.boundscheck(False)
def _dist_mat(np.ndarray[DTYPE_t, ndim=2] V_spec, 
             np.ndarray[DTYPE_t, ndim=2] T_spec):
    """
    Creates the distance matrix between the vocalization spectrogram
    """
    V_spec = np.asfortranarray(V_spec)
    T_spec = np.asfortranarray(T_spec)

    cdef int i
    cdef int j
    cdef int i_max = V_spec.shape[1]
    cdef int j_max = T_spec.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] d = np.zeros([i_max, j_max], dtype=DTYPE)
    spec_len = T_spec.shape[0]

    for i in xrange(i_max):
        for j in xrange(j_max):
            distance(&V_spec[0,i], &T_spec[0,j], &d[i,j], spec_len)

    return d

@cython.boundscheck(False)
def _accumulated_dist(np.ndarray[DTYPE_t, ndim=2] d):
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
    cdef np.ndarray[DTYPE_t, ndim=2, mode = 'c'] D = np.zeros([i_max,j_max], dtype=DTYPE)
    cdef np.ndarray[int, ndim=2, mode = 'c'] T = np.zeros([i_max,j_max], dtype=np.intc)

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
    local_min, = np.where((last_col[1:-1] < last_col[2:]) &
                               (last_col[1:-1] < last_col[:-2]))

    local_min += 1

    if last_col[0] < last_col[1]:
        local_min = np.append(local_min, 0)
    
    if last_col[-1] < last_col[-2]:
        local_min = np.append(local_min, len(last_col) - 1)    
    
    local_min = np.array([lmin for lmin in np.argsort(last_col) if lmin in local_min])

    #convert to index of last_col to index of D
    endpoints = local_min + (D.shape[1]/2)

    return endpoints
    


def _backtrack(int i_end, np.ndarray[int,ndim=2] T):

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
    path = np.zeros((T.shape[1], 2), dtype=int)
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

    return np.flipud(path[:idx,:])



def _interpolate_path(path, d):

    """
    Parameters:
 
    path - indices of path as returned by "backtrack"
    d - distnace matrix (returned by "dist_mat")

    Returns:
    interp_path - array of indices of vocalization spectrogram
    (since interpolation ensures template  step size of 1)
    
    """


    interp_path = np.array([path[0,0]],dtype=int)

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
                i_new =  i - np.argmin([d[i, j-1], d[i-1, j-1]])
            if i_step == 2:
                i_new = i-1
            interp_path = np.append(interp_path, i_new) 
        
        interp_path = np.append(interp_path, i)

    return interp_path
                                

@cython.profile(True)
def _get_paths(D, T, d,  t_0=0):
      
    endpoints = _get_endpoints(D)    
    paths = np.zeros((endpoints.shape[0], D.shape[1]), dtype=int)
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
            i = np.extract(mid_path_bool, P)
            
            T[i,:] = -1
            
    paths=np.delete(paths,np.s_[counter:],0)

    return paths


def find_matches(vocalization, template, fs_voc, fs_temp):
      
    """
    Finds potential matches to a template motif in a recorded vocalization.
    The matches are taken from the local minima last column of the minimum 
    accumulated distance matrix, and are ranked from the smallest distance
    to the template to the largest.  

    Parameters:
 
    vocalization - array containing the sampled data of the recording to be analyzed
    template - array containing the sampled data of the template motif
    fs_voc - sampling rate of the vocalization
    fs_temp - sampling rate of the template

    Returns three lists, each containing one entry for each match
    
    sampled_data - a list of the segments of samlped data where each motif match
                   occurs

    spectrograms - a list of the temporally warped spectrograms of each match 
                    
    dtw_paths - a list in which each entry is a  ector containing the indices 
                of the columns of the vocalization spectrogram used in the temporally
                warped spectrograms
 
    """

        
    T_spec = process_recording(template, fs_temp)
    V_spec = process_recording(vocalization, fs_voc)
    d=_dist_mat(V_spec, T_spec)
    D,T=_accumulated_dist(d)

    dtw_paths=_get_paths(D, T, d)

    spectrograms = [V_spec[:,p] for p in dtw_paths]
    
    spec_step = int(tstep*fs_voc)
    sampled_data = [vocalization[p[0]*spec_step-fft_res:p[-1]*spec_step+fft_res]
                     for p in dtw_paths]
                   
    return sampled_data, spectrograms, dtw_paths
       


    


