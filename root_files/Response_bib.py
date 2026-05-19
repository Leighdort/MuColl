#Bib response and resolution
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

angle = [15, 85]
particle = [11, 211]
for pid in particle:
    for a in angle: 
        #Open up the file
        #Find resolution
        #Graph gaussian
        file = uproot.open(f"/users/rldohert/work/mucoll/mucoll-slurm/files/matched_clusters_pid{pid}_E50_theta{a}_software_on.root")
        tree = file["MatchedClusters"]
        arrays = tree.arrays(library="np")
        matched_energy = arrays["matched_energy"]
        mc_energy = arrays["mc_energy"]
        response = matched_energy / mc_energy 
        
        q16, q84 = np.percentile(response, [16, 84])
        median = np.median(response)
        sigma = (q84 - q16)/2
        resolution = sigma / median
        #mc_low.append(response-q16)
        #mc_high.append(q84-response)
        #mc_median.append(response)
        
        bins = 60
        plt.hist(response, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Response with Bib")
        plt.ylabel(f"Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.2f}\nRes = {resolution:.3f}"
        )
        plt.title(f"Response for pid: {pid}, at 50 GeV for angle {a} Software On")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"response_{pid}_{a}_softwareon.pdf")
        plt.close()
        