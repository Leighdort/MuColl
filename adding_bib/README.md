This folder includes scripts for adding bib using this immage docker://ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:main. For this to run nicely, it will use the cached image.

I currently run 10,000 events full from simulation to reconstruction in sets of 10 which takes ~ 2 minutes per batch job. I use submit_scan.py and full_chain.sh as you might be used to. 
In all of my files I hard code my paths in, so every path should be checked/changed... 
I can elaborate here if needed.

Bib starts once you have gen, sim, reco, and digi files written. They should have only 10 events each, or you might need fewer if dealing with large pt or complicated processes. 
You can use sbatch submit_bib3.sh to submit the bib batch job. You will need to change what you mount to, and the script path. 
Something to point out is the number of events is encoded in the array of the bash codes. Right now it is set from 0-999 (or 10,000 events).

The run_bib.sh is what actually does the job. Here I have the exact paths written into output, input, and the final name of the files. 
Run_bib.sh currently hardcoded to only add bib to one set of parameters in one folder at a time. This can be changed. However, mine currently does not do that as I hit the oscar batch limit at 1000.
Therefore, be careful to currectly writte the paths and what you want the final datasets to be called.

Another sidenote, currently logs/error appears both where the run_bib.sh file is and in the data directory. This can likely be changed, but I haven't gotten to it yet as it currently doens't bother me.
