import argparse
import os
import sys
import ewave
import h5py
from birdwerdz import find_matches
from scipy.cluster.vq import kmeans, vq
import numpy as np
import arf

def main():

        p = argparse.ArgumentParser(prog="classify", 
                               description="""Finds potential instances of given motif and clusters them into groups for further analysis.""")
        p.add_argument("recordings", help="""Either an arf(hdf5) file or a list of wave files containing the recordings to be analyzed""",nargs='+')
        p.add_argument("-t", "--template", help="""Recording of motif to be identified.  May be either a path to the recording dataset within the arf file or a wave file.""", required=True)
        p.add_argument("-o", "--output-name", help="""Name of output hdf5 file. Must be given for wave recordings.""", required=True)
        p.add_argument("-d", "--dataset", help ="""Name of recording dataset within each group in the arf file""")    
        p.add_argument("-c", "--clusters", help="""Number of clusters to use""", default = 10, type = int)
# p.add_argument("-r", "--record-mode", default=60, help="""Analyze arf(hdf5) file that is being written to.  The argument is the number of minutes for the program to run before clustering the matches and exiting. """)

        options = p.parse_args()

        print options.recordings

        with h5py.File(options.output_name, 'w-') as out:

                wavfiles=[]
        
                nonexistent=[]
                unopenable=[]       

                arf_given = None
                RecordingsError=SystemExit("ERROR: Positional argument must be an arf/hdf5 file or a list of wave files")
                for filename in options.recordings:
                        if not os.path.exists(filename):
                            nonexistent.append(filename)
                            continue
                        try:
                            (root,ext) = os.path.splitext(filename)
                            if ext in ('.arf','.hdf5'):

                                if not options.dataset:
                                    sys.exit("ERROR: Must specify recording dataset")

                                if arf_given:
                                    raise RecordingsError 
                                else:
                                    arf_given=True

                                # creating output hdf5/arf file                                      
                                with h5py.File(filename,'r') as src:
                                    for entry in src.values():
                                        if isinstance(entry, h5py.Group) and options.dataset in entry.keys(): 
                                            out.create_group(entry.name)

                                for entry in out.values():
                                        entry[options.dataset] = h5py.ExternalLink(filename, entry.name + '/' + options.dataset)

                            else: # not hdf5/arf file
                                if arf_given: 
                                    raise RecordingsError

                                wavfiles.append(ewave.open(filename))
                                out.create_group(root)
                                out[root].attrs['recording_source']=os.path.abspath(filename)
                        except IOError: 
                                unopenable.append(filename)

                if nonexistent or unopenable:
                    for filename in nonexistent:
                            print "ERROR: %s does not exist" %(filename)
                    for filename in unopenable:
                            print "ERROR: Could not open %s" %(filename)
                    sys.exit()


                # Copying template
                try:
                        if os.path.splitext(options.template)[1] =='.wav':
                                with ewave.open(options.template) as w:
                                        template=w.read()
                                        fs_temp=w.sampling_rate
                                        out.create_dataset('template', data=template)
                                        out['template'].attrs['sampling_rate']=fs_temp
                        else:    
                                with h5py.File(options.recordings[0],'r') as src:
                                        out.copy(src[options.template], 'template')
                                        template=out['template']
                                        if 'sampling_rate' not in out['template'].attrs.keys():
                                                out['template'].attrs['sampling_rate'] = 20000

                                        fs_temp=out['template'].attrs['sampling_rate']

                except Exception as e:
                        sys.exit("ERROR: Could not open template recording")
                        
                         
                #finding matches
                for idx,entry in enumerate(out.values()):
                        
                        if not isinstance(entry, h5py.Group): continue
                        if arf_given:
                                vocalization=entry[options.dataset]
                                fs_voc = entry[options.dataset].attrs['sampling_rate']
                        else:
                                vocalization = wavfiles[idx].read()
                                fs_voc = wavfiles[idx].sampling_rate

                        sampled_data, spectrograms, dtw_paths = find_matches(
                                vocalization, template, fs_voc, fs_temp)
                        
                        #resizing sampled data so it can be put in array
                        max_motif_len = max(len(motif) for motif in sampled_data)                        
                        sampled_data = [np.resize(motif, max_motif_len)
                                        for motif in sampled_data]

                        #putting data in rec_array and saving in hdf5 dataset
                        spec_shape = spectrograms[0].shape
                        path_shape = dtw_paths[0].shape
                        zipped_motifs = zip(sampled_data, spectrograms, dtw_paths)
                        motif_rec = np.array(zipped_motifs,
                                             dtype=[('sampled_data', float, (max_motif_len,)),
                                                    ('spectrogram', float, spec_shape),
                                                    ('dtw_path', int, path_shape)])             
                        entry.create_dataset('motifs', data=motif_rec)
                        print "Found matches for %s" %(entry.name)


                #clustering
                print "Clustering..."

                n_motifs=sum(e['motifs'].shape[0] for e in out.values() 
                             if isinstance(e,h5py.Group) and 'motifs' in e.keys())
                
                #amp_vectors = np.zeros((n_motifs, spec_shape[1]))
                all_spectrograms = np.zeros((n_motifs,) + spec_shape)
                k=0
                for entry in out.values():
                        if not isinstance(entry,h5py.Group) or 'motifs' not in entry.keys(): continue
                        for m in entry['motifs']:
                                all_spectrograms[k,:,:] = m['spectrogram']
                                k+=1

                amp_vectors = all_spectrograms.sum(1)
                centroids,_ = kmeans(amp_vectors, options.clusters)
                id,_ = vq(amp_vectors, centroids)
                mean_spectrograms = np.zeros((options.clusters,) + spec_shape)
                for i in xrange(options.clusters):
                    mean_spectrograms[i,:,:] = all_spectrograms[id==i,:,:].mean(0)

                out.create_dataset('cluster_mean_spectrograms',
                                   data = mean_spectrograms)

if __name__== "__main__":
        main()

                  
