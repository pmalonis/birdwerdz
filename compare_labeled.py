import h5py
import numpy as np
from scipy.cluster.vq import vq
import lbl
from cluster import process_recording
motif_clusters=[2]
recording_dataset='ra16s'
arf=h5py.File('ra16.arf','a')


labeled = [entry for entry in (arf.values()[:154]+arf.values()[155:]) if 'dtw_paths' in entry.keys() and 'lbl' in entry.keys() and len(lbl.find_seq(entry['lbl'], 'abcdefg')) > 0]
labeled_idx = [i for i,entry in enumerate(arf.values()[:154]+arf.values()[155:]) if 'dtw_paths' in entry.keys() and 'lbl' in entry.keys() and len(lbl.find_seq(entry['lbl'], 'abcdefg')) > 0]
# # print sum([lbl.find_seq(entry['lbl'], 'abcdefg').shape[0] for entry in (arf.values()[:154] + arf.values()[155:]) if 'lbl' in entry.keys()])


# for entry in labeled:
#     paths = np.array(entry['dtw_paths'],dtype=int)
#     centroids = arf['model']['centroids'][...]
#     V_spec=process_recording(entry[recording_dataset],entry.attrs['sampling_rate'])
#     amp_vec = np.zeros(paths.shape)
#     for i, path in enumerate(paths):
#         amp_vec[i,:] = V_spec[:,path].sum(0)

#     id,_ = vq(amp_vec, centroids)
    
#     starts = np.array([])
#     for n in motif_clusters:        
#         starts = np.append(starts,paths[id==n, 0])

#     if 'motif_starts' in entry.keys():
#         del entry['motif_starts']

#     entry.create_dataset("motif_starts", data=starts)

def find_error(diff_thresh=.1):
    hits, false_pos, false_neg = [0]*3
    h_specs=[]
    fp_specs=[]
    fn_specs=[]
    
    fp_loc=[]
    fn_loc=[]
    h_loc=[]
    for i,entry in enumerate(labeled):
        lbl_starts = lbl.find_seq(entry['lbl'],'abcdefg')[:,0]
        V_spec=process_recording(entry['ra16s'],entry.attrs['sampling_rate'])
        for k,start in enumerate(entry['motif_starts'][:]):           
            paths = np.array(entry['dtw_paths'], dtype=int)
            p = paths[paths[:,0]==start,:][0,:]
            if np.min(np.abs(lbl_starts-start/1000.0)) < diff_thresh:
                lbl_starts=np.delete(lbl_starts, np.argmin(np.abs(lbl_starts-start/1000.0)))
                hits +=1
                h_specs.append(V_spec[:,p])
                h_loc.append((labeled_idx[i],start))
            else:
                false_pos += 1                                             
                fp_specs.append(V_spec[:,p])
                fp_loc.append((labeled_idx[i],start))

            if len(lbl_starts)==0:
                remaining=len(entry['motif_starts'][:])-(k+1)
                if remaining:
                    false_pos+=remaining
                    for s in entry['motif_starts'][k+1:]:
                        fp_loc.append((labeled_idx[i],s))
                        p = paths[paths[:,0]==s,:][0,:]
                        fp_specs.append(V_spec[:,p])               
                break

        if len(lbl_starts)>0:
            false_neg+=len(lbl_starts)
            for ls in lbl_starts:
                fn_specs.append(V_spec[:,np.round(ls*1000):np.round(ls*1000)+951])
                fn_loc.append((labeled_idx[i],ls))
                
    return (hits, false_pos, false_neg), fn_specs, fp_specs, h_specs


