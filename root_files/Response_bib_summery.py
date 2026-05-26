#Response_bib.py just for making summery graphs
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
    #Lets do 15 and 85
    #Then for soft and nosoft bib 
    median_soft_15 = {"bib": [], "nobib": []}
    median_nosoft_15 = {"bib": [], "nobib": []}
    low_soft_15 = {"bib": [], "nobib": []}
    low_nosoft_15 = {"bib": [], "nobib": []}
    high_soft_15 = {"bib": [], "nobib": []}
    high_nosoft_15 = {"bib": [], "nobib": []}
    res_soft_15 = {"bib": [], "nobib": []}
    res_nosoft_15 = {"bib": [], "nobib": []}
    median_soft_85 = {"bib": [], "nobib": []}
    median_nosoft_85 = {"bib": [], "nobib": []}
    low_soft_85 = {"bib": [], "nobib": []}
    low_nosoft_85 = {"bib": [], "nobib": []}
    high_soft_85 = {"bib": [], "nobib": []}
    high_nosoft_85 = {"bib": [], "nobib": []}
    res_soft_85 = {"bib": [], "nobib": []}
    res_nosoft_85 = {"bib": [], "nobib": []}
    for a in angle: 
        #Open up the file
        #Find resolution
        #Graph gaussian
        file = uproot.open(f"/users/rldohert/work/mucoll/mucoll-slurm/files/matched_clusters_pid{pid}_E50_theta{a}_software_off_nobib.root")
        tree = file["MatchedClusters"]
        arrays = tree.arrays(library="np")
        matched_energy = arrays["matched_energy"]
        mc_energy = arrays["mc_energy"]
        mc_energy = np.asarray(mc_energy).reshape(-1)
        matched_energy = np.asarray(matched_energy).reshape(-1)
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0])  # what does one entry look like?
        #matched_energy = np.array(matched_energy, dtype=float)
        #mc_energy = np.array(mc_energy, dtype=float)
        response = matched_energy / mc_energy 
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0]) 
        q16, q84 = np.percentile(response, [16, 84])
        median = np.median(response)
        sigma = (q84 - q16)/2
        resolution = sigma / median
        if a == 15:
            median_nosoft_15["nobib"].append(median)
            low_nosoft_15["nobib"].append(median-q16)
            high_nosoft_15["nobib"].append(q84-median)
            res_nosoft_15["nobib"].append(resolution)
        elif a == 85:
            median_nosoft_85["nobib"].append(median)
            low_nosoft_85["nobib"].append(median-q16)
            high_nosoft_85["nobib"].append(q84-median)
            res_nosoft_85["nobib"].append(resolution)

        
        file = uproot.open(f"matched_clusters_pid{pid}_E50_theta{a}_software_on.root")
        tree = file["MatchedClusters"]
        arrays = tree.arrays(library="np")
        matched_energy = arrays["matched_energy"]
        mc_energy = arrays["mc_energy"]
        mc_energy = np.asarray(mc_energy).reshape(-1)
        matched_energy = np.asarray(matched_energy).reshape(-1)
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0])  # what does one entry look like?
        #matched_energy = np.array(matched_energy, dtype=float)
        #mc_energy = np.array(mc_energy, dtype=float)
        response = matched_energy / mc_energy 
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0]) 
        q16, q84 = np.percentile(response, [16, 84])
        median = np.median(response)
        sigma = (q84 - q16)/2
        resolution = sigma / median
        if a == 15:
            median_soft_15["bib"].append(median)
            low_soft_15["bib"].append(median-q16)
            high_soft_15["bib"].append(q84-median)
            res_soft_15["bib"].append(resolution)
        elif a == 85:
            median_soft_85["bib"].append(median)
            low_soft_85["bib"].append(median-q16)
            high_soft_85["bib"].append(q84-median)
            res_soft_85["bib"].append(resolution)

        #Open up the file
        #Find resolution
        #Graph gaussian
        file = uproot.open(f"/users/rldohert/work/mucoll/mucoll-slurm/files/matched_clusters_pid{pid}_E50_theta{a}_software_on_nobib.root")
        tree = file["MatchedClusters"]
        arrays = tree.arrays(library="np")
        matched_energy = arrays["matched_energy"]
        mc_energy = arrays["mc_energy"]
        mc_energy = np.asarray(mc_energy).reshape(-1)
        matched_energy = np.asarray(matched_energy).reshape(-1)
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0])  # what does one entry look like?
        #matched_energy = np.array(matched_energy, dtype=float)
        #mc_energy = np.array(mc_energy, dtype=float)
        response = matched_energy / mc_energy 
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0]) 
        q16, q84 = np.percentile(response, [16, 84])
        median = np.median(response)
        sigma = (q84 - q16)/2
        resolution = sigma / median
        if a == 15:
            median_soft_15["nobib"].append(median)
            low_soft_15["nobib"].append(median-q16)
            high_soft_15["nobib"].append(q84-median)
            res_soft_15["nobib"].append(resolution)
        elif a == 85:
            median_soft_85["nobib"].append(median)
            low_soft_85["nobib"].append(median-q16)
            high_soft_85["nobib"].append(q84-median)
            res_soft_85["nobib"].append(resolution)

        
        file = uproot.open(f"matched_clusters_pid{pid}_E50_theta{a}_software_off.root")
        tree = file["MatchedClusters"]
        arrays = tree.arrays(library="np")
        matched_energy = arrays["matched_energy"]
        mc_energy = arrays["mc_energy"]
        mc_energy = np.asarray(mc_energy).reshape(-1)
        matched_energy = np.asarray(matched_energy).reshape(-1)
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0])  # what does one entry look like?
        #matched_energy = np.array(matched_energy, dtype=float)
        #mc_energy = np.array(mc_energy, dtype=float)
        response = matched_energy / mc_energy 
        print(type(matched_energy))
        print(matched_energy.shape)
        print(matched_energy.dtype)
        print(matched_energy[0]) 
        q16, q84 = np.percentile(response, [16, 84])
        median = np.median(response)
        sigma = (q84 - q16)/2
        resolution = sigma / median
        if a == 15:
            median_nosoft_15["bib"].append(median)
            low_nosoft_15["bib"].append(median-q16)
            high_nosoft_15["bib"].append(q84-median)
            res_nosoft_15["bib"].append(resolution)
        elif a == 85:
            median_nosoft_85["bib"].append(median)
            low_nosoft_85["bib"].append(median-q16)
            high_nosoft_85["bib"].append(q84-median)
            res_nosoft_85["bib"].append(resolution)

        
        '''
        lo, hi = 0, 2
        bins=60
        plt.hist(response, bins=bins, range=(lo, hi), edgecolor='none', color='steelblue')
        plt.yscale('log')
        plt.ylim(bottom=1)  # start y-axis at 1 so empty bins don't show as 0 on log scale  
        #bins = 60
        #plt.hist(response, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Response without Bib")
        plt.ylabel(f"Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.2f}\nRes = {resolution:.3f}"
        )
        plt.title(f"Response for pid: {pid}, at 50 GeV for angle {a} Software Off")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"response_{pid}_{a}_softwareoff.pdf")
        plt.close()
        '''
   

    plt.errorbar(
        angle,
        [median_soft_15["bib"][0], median_soft_85["bib"][0]],
        yerr=[
            [low_soft_15["bib"][0], low_soft_85["bib"][0]],
            [high_soft_15["bib"][0], high_soft_85["bib"][0]]
        ],
        fmt='o-',
        capsize=4,
        label="Software On + BIB"
    )

    plt.errorbar(
        angle,
        [median_soft_15["nobib"][0], median_soft_85["nobib"][0]],
        yerr=[
            [low_soft_15["nobib"][0], low_soft_85["nobib"][0]],
            [high_soft_15["nobib"][0], high_soft_85["nobib"][0]]
        ],
        fmt='s-',
        capsize=4,
        label="Software On + No BIB"
    )

    plt.errorbar(
        angle,
        [median_nosoft_15["bib"][0], median_nosoft_85["bib"][0]],
        yerr=[
            [low_nosoft_15["bib"][0], low_nosoft_85["bib"][0]],
            [high_nosoft_15["bib"][0], high_nosoft_85["bib"][0]]
        ],
        fmt='^-',
        capsize=4,
        label="Software Off + BIB"
    )

    plt.errorbar(
        angle,
        [median_nosoft_15["nobib"][0], median_nosoft_85["nobib"][0]],
        yerr=[
            [low_nosoft_15["nobib"][0], low_nosoft_85["nobib"][0]],
            [high_nosoft_15["nobib"][0], high_nosoft_85["nobib"][0]]
        ],
        fmt='d-',
        capsize=4,
        label="Software Off + No BIB"
    )

    plt.xlabel("Angle (deg)")
    plt.ylabel("Median Response ")
    plt.title(f"Response for Software On/Off Matched Clusters with/without BIB {pid}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"response_vs_angle_softonoff_bib_{pid}.pdf")
    plt.close()

    plt.plot(
    angle,
    [res_soft_15["bib"][0], res_soft_85["bib"][0]],
    'o-',
    label="Software On + BIB"
    )

    plt.plot(
        angle,
        [res_soft_15["nobib"][0], res_soft_85["nobib"][0]],
        's-',
        label="Software On + No BIB"
    )

    plt.plot(
        angle,
        [res_nosoft_15["bib"][0], res_nosoft_85["bib"][0]],
        '^-',
        label="Software Off + BIB"
    )

    plt.plot(
        angle,
        [res_nosoft_15["nobib"][0], res_nosoft_85["nobib"][0]],
        'd-',
        label="Software Off + No BIB"
    )

    plt.xlabel("Angle (deg)")
    plt.ylabel("Resolution")
    plt.title(f"Resolution for Software On/Off Matched Clusters with/without BIB {pid}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"resolution_vs_angle_softonoff_bib_{pid}.pdf")
    plt.close()
        

'''
"ecal_barrel": np.array([x["ecal_barrel"] for x in matched_clusters], dtype=np.int32),
                "hcal_barrel": np.array([x["hcal_barrel"] for x in matched_clusters], dtype=np.int32),
                "ecal_endcap": np.array([x["ecal_endcap"] for x in matched_clusters], dtype=np.int32),
                "hcal_endcap": np.array([x["hcal_endcap"] for x in matched_clusters], dtype=np.int32),
'''