#!/usr/bin/env python
"""
Functions for analyzing recordings stored in hdf5 files
"""
import ewave
import sys
import h5py
from birdwerdz import dtw
import numpy as np
from scipy.cluster.vq import vq, kmeans
import os
import arf
import matplotlib.pyplot as plt


def classify(recordings, template, output_name, 
             dataset_name='', nclusters=10, tstep=.001,
             win_len=256, noise_cutoff=500, 
             numtaps=101, max_freq=10000):
    """
    Finds potential instances of given motif and clusters them into groups for further analysis

    Parameters
    ----------
    recordings : The name of an arf(hdf5) file containing the recordings to be analyzed. By default, 
                 recording only datasets that have datatype set to 1, or 
                 (ACOUSTIC as per the arf specification) will be analyzed. Alternatively, 
                 you can use the dataset_name argument to specify that name of the dataset within
                 each group that you want analyzed.
    template : Recording of motif to be identified.  May be either a path to the recording dataset within
                the arf file or the name of a wave file
    output_name : Name of output hdf5 file. Must be given for wave recordings
    dataset_name : Name of datasets that consist of audio recodings to be analyzed.
    clusters : Number of clusters to use for k-means clustering
    tstep : Time step of spectrogram in seconds
    win_len : length of window used to compute spectrogram, in samples
    noise_cutoff : cutoff frequency for high pass filter
    numtaps : Number of filter taps
    max_freq : Maximum frequency in the spectrogram to analyze
    """
    try:
        h5py.File(output_name, 'w-').close()
    except IOError:
        sys.exit('Output file %s already exists' %(output_name))
        
    with h5py.File(output_name, 'r+') as out:        
        # creating output hdf5/arf file    
        with h5py.File(recordings,'r') as src:
            recording_names = [] # name of recording datasets
            for entry in src.itervalues():
                if not isinstance(entry, h5py.Group): continue
                try:
                    if dataset_name == '':
                        # finds first dataset in entry.values() with datatype set to 1 (acoustic) 
                        dataset = (dset for dset in entry.values() if
                                   ('datatype' in dset.attrs.iterkeys() and
                                    dset.attrs['datatype'] == 1)).next()
                    else:
                        dataset = entry[dataset_name]
                except StopIteration:
                    continue

                recording_names.append(dataset.name.split('/')[-1])
                out.create_group(entry.name)

        for entry, dataset in zip(out.itervalues(), recording_names):
            entry[dataset] = h5py.ExternalLink(recordings, entry.name + '/' + dataset)

        # Copying template
        try:
            with h5py.File(recordings,'r') as src:
                out.copy(src[template], 'template')
                template=out['template']
                if 'sampling_rate' not in out['template'].attrs.iterkeys():
                    out['template'].attrs['sampling_rate'] = 20000
                fs_temp=out['template'].attrs['sampling_rate']
        except Exception as e:
            print("Could not open template recording")
            raise e

        #finding matches
        for idx,entry in enumerate(out.itervalues()):
            if not isinstance(entry, h5py.Group): continue
        
            vocalization = entry.values()[0]
            fs_voc = vocalization.attrs['sampling_rate']                

            sampled_data, spectrogram, dtw_paths,_ = dtw.find_matches(
                    vocalization, template, fs_voc, fs_temp, all_spectrograms=False)

            #resizing sampled data so it can be put in array
            max_motif_len = max(len(motif) for motif in sampled_data)                        
            sampled_data = [np.resize(motif, max_motif_len)
                            for motif in sampled_data]

            #putting data in rec_array and saving in hdf5 dataset
            freq_size = spectrogram.shape[0] #size of frequency axis of spectrogram
            path_length = dtw_paths.shape[1]
            spec_shape = (freq_size, dtw_paths.shape[1])
            dtype=[('motif_interval', float, (max_motif_len,)),
                   ('spectrogram', float, spec_shape),
                   ('dtw_path', int, (path_length,))]
            nmatches = len(dtw_paths)
            entry.create_dataset('motifs', shape=(nmatches,), dtype=dtype)
            for i in xrange(nmatches):
                entry['motifs'][i] = (sampled_data[i], 
                                      spectrogram[:, dtw_paths[i]], 
                                      dtw_paths[i])
            if nmatches > 0:
                entry['motifs'].attrs['win_len'] = win_len
                entry['motifs'].attrs['noise_cutoff'] = noise_cutoff
                entry['motifs'].attrs['tstep'] = tstep
                entry['motifs'].attrs['numtaps'] = numtaps
                

            print("Found matches for %s" %(entry.name))

    #clustering
    # print("Clustering...")
    # cluster(output_name, nclusters)
        

# def cluster(file, nclusters=10):
#     """
#     Clusters potential motifs that are outputted by the "classify" function.

#     Parameters
#     ----------
#     file : An hdf5 file containing motif matches as generated by birdwerdz.hdf.classify
#     clusters : Number of clusters to use
#     """
#     with h5py.File(file, 'r+') as f:
#         #import pdb;pdb.set_trace()
#         n_motifs=sum(e['motifs'].size for e in f.itervalues() 
#                      if isinstance(e,h5py.Group) and 'motifs' in e.keys())

#         #finding spectrogram shape
#         spec_shape = None
#         for entry in f.itervalues():
#             if (isinstance(entry,h5py.Group)
#                 and 'motifs' in entry.keys()
#                 and entry['motifs'].size):
#                     spec_shape = entry['motifs']['spectrogram'][0].shape
#                     break

#         if spec_shape is None:
#             return

#         all_spectrograms = np.zeros((n_motifs,) + spec_shape)
#         k=0
#         for entry in f.itervalues():
#             if (not isinstance(entry,h5py.Group)
#                 or 'motifs' not in entry.keys()): continue
#             n = entry['motifs'].size
#             all_spectrograms[k:k+n,:,:] = entry['motifs']['spectrogram']
#             k += n

#         id, mean_spectrograms = dtw.cluster_motifs(all_spectrograms,
#                                                    nclusters=nclusters)

#         cluster_path = 'cluster_mean_spectrograms'
#         if cluster_path in f.keys():
#             del f[cluster_path]

#         f.create_dataset(cluster_path, data = mean_spectrograms)

def cluster(file, nclusters=10):
    """
    Clusters potential motifs that are outputted by the "classify" function.

    Parameters
    ----------
    file : An hdf5 file containing motif matches as generated by birdwerdz.hdf.classify
    clusters : Number of clusters to use
    """
    with h5py.File(file, 'r+') as f:
        #import pdb;pdb.set_trace()
        n_motifs=sum(e['motifs'].size for e in f.itervalues() 
                     if isinstance(e,h5py.Group) and 'motifs' in e.keys())

        #finding spectrogram shape
        spec_shape = None
        for entry in f.itervalues():
            if (isinstance(entry,h5py.Group)
                and 'motifs' in entry.keys()
                and entry['motifs'].size):
                    spec_shape = entry['motifs']['spectrogram'][0].shape
                    break

        # if spec_shape is None:
        #     return

        all_amp_vectors = np.zeros((n_motifs, + spec_shape[1]))
        k=0
        for entry in f.itervalues():
            if (not isinstance(entry,h5py.Group)
                or 'motifs' not in entry.keys()): continue          
            n = entry['motifs'].size
            all_amp_vectors[k:k+n,:] = entry['motifs']['spectrogram'].sum(1)
            k += n
            
        centroids,_ = kmeans(all_amp_vectors, nclusters)
        id,_ = vq(all_amp_vectors, centroids)
        
        mean_spectrograms = np.zeros((nclusters,) + spec_shape)
        i = 0
        for entry in f.itervalues():
            if (not isinstance(entry,h5py.Group)
                or 'motifs' not in entry.keys()): continue
            for spec in entry['motifs']['spectrogram']:
                mean_spectrograms[id[i],:,:] *= i/(i+1)
                mean_spectrograms[id[i],:,:] += spec/(i+1)
                i += 1

        cluster_path = 'cluster_mean_spectrograms'

        if cluster_path in f.keys():
            del f[cluster_path]

        f.create_dataset(cluster_path, data=mean_spectrograms)

        # counting number of examples in each cluster and saving as attribute
        cluster_sizes,_ = np.histogram(id, range(nclusters+1))
        f[cluster_path].attrs['cluster_sizes'] = cluster_sizes
                
def plot(motif_file):
    """
    Plots the mean spectrogram of each motif cluster.

    Parameters
    ----------
    motif_file : An hdf5 file containing clustered motif 
    matches as generated by birdwerdz.hdf.classify
    """
    def template_yaxis(y_ax):
        y_ax.label.set_rotation(0)
        y_ax.label.set_size(16)

        y_ax.set_ticks([])
        y_ax.set_label_text('Template')
        y_ax.label.set_horizontalalignment('right')

    def cluster_yaxis(y_ax, label, txt_size=20, y_pos=.25, alignment='right'):
        y_ax.label.set_rotation(0)
        y_ax.label.set_size(txt_size)
        y_ax.label.set_y(y_pos)
        y_ax.set_ticks([])
        y_ax.set_label_text(label)
        y_ax.label.set_horizontalalignment(alignment)
    
    with h5py.File(motif_file,'r+') as motifs_hdf:            

        cluster_path = 'cluster_mean_spectrograms'
        template_path = 'template'
        nplots = motifs_hdf[cluster_path].shape[0] + 2
        T_spec = dtw.process_recording(motifs_hdf['template'], 
                                       motifs_hdf[template_path].attrs['sampling_rate'])

        f=plt.figure(figsize=(10,10))

        sp=f.add_subplot(nplots,1,1)
        sp.imshow(T_spec, origin='lower', aspect='auto')
        sp.set_title('Template')
        template_yaxis(sp.yaxis)
        sp.xaxis.set_visible(False)

        for i,spectrogram in enumerate(motifs_hdf[cluster_path]):
            if i == 0:
                spec_min = spectrogram.min()
                spec_max = spectrogram.max()

            spec_sp = f.add_subplot(nplots, 1, i+3)
            spec_sp.imshow(spectrogram, origin='lower', aspect='auto',
                           vmin=spec_min, vmax=spec_max)

            # transparent axis that will be used to print the number of examples 
            # on the right side of the plot
            right_axis_sp = f.add_subplot(nplots, 1, i+3, 
                                          sharex=sp, frameon=False) 

            right_axis_sp.yaxis.tick_right()
            right_axis_sp.yaxis.set_label_position('right')
            n_examples = motifs_hdf[cluster_path].attrs['cluster_sizes'][i]
            cluster_yaxis(right_axis_sp.yaxis, 'n=%d'%n_examples, 
                          txt_size=12, y_pos=.75, alignment='left')
            cluster_yaxis(spec_sp.yaxis, i)
            if i == 0:
                spec_sp.set_title('Cluster Mean Spectrograms')
            if i+3 < nplots:
                spec_sp.xaxis.set_visible(False)

        sp.xaxis.set_label_text('Time (ms)')
        plt.show()
            

def select(file, output, clusters=None):
    """
    Select clusters containing real motifs and discard the rest

    Parameters
    ----------
    file : An hdf5 file containing clustered motif matches as generated by birdwerdz.hdf.classify
    output : Name of output file which will contain only motifs from selected
             clusters.  If same as input file, will delete motifs from the file
    clusters : Clusters to select 

    """
    if file == output:
        mode = 'r+'
    else:
        mode = 'w-'
    with h5py.File(output, mode) as out:
        if file != output:
            with h5py.File(file, 'r+') as src:
                for entry in src.values():
                    out['/'].copy(entry,entry.name)
        for entry in out.values():
            if not isinstance(entry,h5py.Group) or 'motifs' not in entry.keys():
                continue

            amp_vecs= entry['motifs']['spectrogram'].sum(1) 

            cluster_path = 'cluster_mean_spectrograms'
            id,_ = vq(amp_vecs, out[cluster_path][:].sum(1))

            new_motifs=np.delete(entry['motifs'], np.where(
                [i not in clusters for i in id])[0])

            del entry['motifs']
            entry.create_dataset('motifs',data=new_motifs)
        

def label(motif_file, recordings, label, 
          label_name='auto_lbl'): # 
    """
    Creates label entry from motif matches
    Parameters
    ----------
    motif_file : An hdf5 file containing clustered motif matches as generated by birdwerdz.hdf.classify
    recordings : The name of the arf(hdf5) file containings the raw recordings to be labeled
    label : Path to template label in recordings file
    label_name : Name of the label datasets to be made
    """
    #todo: convert starts and stops to spectrogram index
    with h5py.File(recordings, 'r+') as rec_file:
        template_lbl = rec_file[label]
        units = template_lbl.attrs['units']
        unit_args = {'units' : ['', units, units]}
        if units == 's':
            spec_res = tstep
        elif units  == 'ms':
            spec_res = tstep * 1000 
        elif units == 'samples':
            sr = template_lbl.attrs['sampling_rate']
            spec_res = tstep*float(sr)
            unit_args['sampling_rate'] = sr

        with h5py.File(motif_file, 'r+') as motif:
            #getting length of template spectrogram
            template_len = None
            for entry in motif.itervalues():
                if (isinstance(entry,h5py.Group)
                    and 'motifs' in entry.keys()
                    and entry['motifs'].size):
                    template_len = entry['motifs']['dtw_path'].shape[0]
                    break
            
            if template_len is None:
                return
            
            start_idx = [max(0,int(start/spec_res)) for start in template_lbl['start']]
            stop_idx = [min(int(stop/spec_res),template_len) for stop in template_lbl['stop']]
            names = template_lbl['name']
            for entry in motif.values():
                if (not isinstance(entry,h5py.Group) or 'motifs' not in entry.keys()
                    or not entry['motifs'].size):
                    continue
                    
                dtype = [('name', 'a' + str(max([len(x) for x in names]))),
                         ('start', float), ('stop', float)]               
                lbl = np.array([(l, m['dtw_path'][b]*spec_res, m['dtw_path'][e]*spec_res) for m in entry['motifs'] for l,b,e in zip(names, start_idx, stop_idx)], dtype=dtype)
                arf.create_dataset(rec_file[entry.name], label_name,
                                   data=lbl, maxshape=(None,), datatype=2002,**unit_args)
