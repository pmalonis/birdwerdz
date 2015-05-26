# birdwerdz

Birdwerdz is an automated birdsong recongnitzion tool based on an algorithm published by Fantana 
and Kozhevnikov (2014). 

### Requirements
Birdwerdz is a Python 2 package which requires the packages Cython, numpy, scipy, matplotlib, and [arf](https://github.com/margoliashlab/arf). The package incorporates a Cython extension module which should work cross platforms but has only been tested on Linux.

### Installation
To install, clone this repository, cd into the repository folder, and enter this command as root:

    python setup.py install

### Overview
The algorithm uses DTW to find potential matches it a recording to a given template.  These 
matches represent local minima in the space of mappings between the template recording 
and the recording to be analyzed.  This set of matches contains all of the examples of the 
template vocalization, as well as many false positives.  In the second step of the algorithm a 
simple cluster analysis is performed on the potential matches in order to obtain the final 
results.
  
### Command-line interface


-----------
Reference:

Fantana, A. L., & Kozhevnikov, A. (2014). Finding motifs in birdsong data in the presence of acoustic noise and temporal jitter. Behavioral Neuroscience, 128(2), 228–236. http://doi.org/10.1037/a0035985
