import numpy as np
from scipy.cluster.vq import kmeans
import matplotlib.pyplot as plt
import h5py
import lbl


def main():

        p = argparse.ArgumentParser(prog="label"
                                    description="Creates lbl array from motif matches")
        p.add_argument("motif-file", help="""An arf(hdf5) file generated by classify.py""", nargs='+')
        p.add_argument("-l, --labels", help="""Syllable to use """, nargs='+',required=True)
        p.add_argument("-b, --beginnings", help="""Time in the template (in ms) at which each syllable starts""", nargs='+',required=True)
        p.add_argument("-e, --endings", help="""Time in the template (in ms) at which each syllable stops""", nargs='+')
        options = p.parse_args()
        
        if len(options.labels) != len(options.beginnings):
            sys.exit("ERROR: The number of labels and the number of times given must be the same")
            
        if not options.endings:
            options.endings = [None]*len(options.beginnings)
            
        with h5py.File(options.motif_file, 'r+'):
            for entry in arf.values():
                if not isinstance(entry,h5py.Group) or 'motifs' not in entry.keys():
                    continue
                
                dtype = [('name', 'a' + str(max([len(x) for x in options.labels]))),
                         ('start', float), ('stop', float)]
            
                spec_res = .001
                syllables = [(l, m['dtw_paths'][b]*spec_res, m['dtw_paths'][e]*spec_res) 
                             for m in entry['motifs']
                             for l,b,e in zip(options.labels, 
                                              options.beginnings, 
                                              options.endings)]
                entry.create_dataset('lbl', dtype=dtype,data=syllables)