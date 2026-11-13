#Multiclus_res.pyplot
#This gives you gaussians

#Combine the response graphs w/ the ratio graphs 
#Plot the response color code by ratio yellow = one, blue = 0 

#all in hcal = 0
#all in ecal = 1
#So the number is the count in the ecal
#We want to fill an array with response
#Fill another array with the ecal ratio

#This a test of correlating fraction to where fraction of hits are found

#This is plotting the original and the edited functions
angle2region = {
    15: "Endcap",
    85: "Barrel",
    143: "Transition",
}
region_colors = {
    "Barrel": "tab:orange",
    "Endcap": "tab:blue",
}

calib_styles = {
    "Baseline": "-",
    "4Cal": "--",
    "4CalHitCal": ":",
}

calib_markers = {
    "Baseline": "o",   # circle
    "4Cal": "s",      # square
    "4CalHitCal": "^" # triangle
}

particle_labels = {
    11: "Electron",
    211: "Pion"
}
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
#from scipy.optimize import curve_fit

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
#We would also have to loop through energies here but just wait first 
bounds = {
    211: {  # pion
        15: {2: 0.05, 10: 0.02, 30: 0.01, 50: 0.006},
        85: {2: 0.82, 10: 0.06, 30: 0.02, 50: 0.02},
        143:{2: 0.41, 10: 0.03, 30: 0.01, 50: 0.008},
    },
    11: {   # electron
        15: {2: 0.01, 10: 0.01, 30: 0.01, 50: 0.01},
        85: {2: 1.40, 10: 0.09, 30: 0.06, 50: 0.06},
        143:{2: 0.59, 5: 0.05, 10: 0.04, 15: 0.04, 30: 0.04, 50: 0.04},
    }
}
#energy = [2, 5, 10, 30, 50, 75]
energy = [50]

#energy = [50]
#angles = [15, 85]
angles = [85]
particle = [211]
#energy = [30, 50]
# e 2, 10, 30, 50 
#particle = [2112, 2212, 321]
#particle = [11]
#angles = [15, 85, 140]
#angles = [15, 85]
#angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yes = True
results = {}
results_base = {}
results_4cal = {}
#This is exluding bounds!

#Now we need the other thing too here
#This is Graphing the Full Thing
for pid in particle:
    for e in energy: 
        mc_median= []
        mc_low = []
        mc_high = []
        for a in angles:
            print(f"\n=== Energy {a} Degrees ===")
            fraction = []
            num_events_1_clus = 0
            passes = 0
            #bound = bounds[pid][a][10] #this is not a trial run, og run
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial10.2/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial5/reco_pdg_pdg_pt_pt_theta_theta_nobib.root")
            file = uproot.open(f"/users/rldohert/work/mucoll/mucoll-slurm/files/matched_clusters_pid{pid}_E{e}_theta{a}_TOVA_base.root")
            
            angular_dist = []
            regular_dist = []
            ecal_e_ratio = []
            hcal_e_ratio = []
            ecal_b_ratio = []
            hcal_b_ratio = []
            ecal_ratio = []
            hcal_ratio = []
            ecal_energy_ratio = []
            hcal_energy_ratio = []
            response = []
            events_below_one = []
            tree = file["MatchedClusters"]
            arrays = tree.arrays(library="np")
            #arrays = np.asarray(arrays).reshape(-1)
            #mc_energy = np.asarray(mc_energy).reshape(-1)
            
            #print(len(file["MatchedClusters"]["event"]))
            #mc_energy = np.asarray(arrays["mc_energy"]).reshape(-1)
            mc_energy = arrays["mc_energy"]
            mc_energy = np.asarray(mc_energy).reshape(-1)
            clus_energy = arrays["matched_energy"]
            clus_energy = np.asarray(clus_energy).reshape(-1)
            #clus_energy = np.asarray(arrays["beam_energy"]).reshape(-1)
            ecal_barrel = np.asarray(arrays["ecal_barrel"]).reshape(-1)
            ecal_endcap = np.asarray(arrays["ecal_endcap"]).reshape(-1)
            hcal_barrel = np.asarray(arrays["hcal_barrel"]).reshape(-1)
            hcal_endcap = np.asarray(arrays["hcal_endcap"]).reshape(-1)
            event_num = np.asarray(arrays["event"]).reshape(-1)
            print(len(mc_energy))
            for i in range((len(mc_energy))):
                #print("hi")
                #Let's just right now filter for events with only 1 cluster
                mc_E = mc_energy[i]
                clus_E = clus_energy[i]
                total = ecal_barrel[i] + ecal_endcap[i] + hcal_endcap[i] + hcal_barrel[i]
                ecal_ratio_i = ecal_endcap[i] / total + ecal_barrel[i] / total
                ecal_ratio.append(ecal_ratio_i)
                response.append(clus_E / mc_E)
                #print(clus_E / mc_E)
                #print(clus_E)
                #print(mc_E)
                #print(i)
                if (clus_E / mc_E < 1):
                    events_below_one.append(event_num[i])
            response = np.array(response) #cluster / mx[0] to recalibrate multiply the top number 
            ecal_ratio = np.array(ecal_ratio)
            events_below_one = np.array(events_below_one)
            for j in range(len(events_below_one)):
                print(events_below_one[j])
            if pid not in results_base:
                results_base[pid] = {}
            if a not in results_base[pid]:
                results_base[pid][a] = {
                    "energy": [],
                    "median": [],
                    "low": [],
                    "high": []
                }
            q16, q84 = np.percentile(response, [16, 84])
            median_full = np.median(response)
            results_base[pid][a]["energy"].append(e)
            results_base[pid][a]["median"].append(median_full)
            results_base[pid][a]["low"].append(median_full - q16)
            results_base[pid][a]["high"].append(q84 - median_full)
            bins = np.linspace(0, 2, 60)
            plt.figure(figsize=(7,5))
            median = np.median(response)
            below_50 = []
            above_50 = []
            ecal_low = []
            ecal_high = []
            for i in range(len(response)):
                if ecal_ratio[i] < 0.5:
                    below_50.append(response[i])
                    ecal_low.append(ecal_ratio[i])
                if ecal_ratio[i] >= 0.5:
                    above_50.append(response[i])
                    ecal_high.append(ecal_ratio[i])
            median_below = np.median(below_50)
            median_above = np.median(above_50)
            bins = np.linspace(0, 2, 60)
            below_50 = np.array(below_50)
            above_50 = np.array(above_50)
            ecal_low = np.array(ecal_low)
            ecal_high = np.array(ecal_high)

            plt.hist(below_50, bins=bins, alpha=0.5,
            label='Ecal Hit Fraction < 0.5', edgecolor='black')

            plt.hist(above_50, bins=bins, alpha=0.5,
                    label='Ecal Hit Fraction ≥ 0.5',
                    edgecolor='black')

            plt.axvline(median_below, color = 'royalblue', linestyle='--', linewidth=2,
                        label=f'Below 0.5 median = {median_below:.3f}')

            plt.axvline(median_above, color='orange', linestyle='--', linewidth=2,
                        label=f'Above 0.5 median = {median_above:.3f}')

            plt.axvline(median, color='red', linestyle='--', linewidth=2,
                        label=f"Median = {median:.6f}")

            plt.xlabel("Response (Multi-Cluster)")
            plt.ylabel("Count")
            plt.xlim(0, 2)
            #plt.title(f"Baseline Gaussian (Pion, Barrel, {e} GeV)")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(f"gaussian_TOVA_baseline_{a}_{pid}_{e}_multicluster.pdf")
            plt.close()


'''
for pid in particle:
    plt.figure(figsize=(7,5))
    for a in angles:
        E = results[pid][a]["energy"]
        med = results[pid][a]["median"]
        low = results[pid][a]["low"]
        high = results[pid][a]["high"]
        E_base = results_base[pid][a]["energy"]
        med_base = results_base[pid][a]["median"]
        low_base = results_base[pid][a]["low"]
        high_base = results_base[pid][a]["high"]
        E_4cal = results_4cal[pid][a]["energy"]
        med_4cal = results_4cal[pid][a]["median"]
        low_4cal = results_4cal[pid][a]["low"]
        high_4cal = results_4cal[pid][a]["high"]
        region = angle2region[a]
        color = region_colors[region]
        plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
        
        plt.errorbar(
            E_base,
            med_base,
            yerr=[low_base, high_base],
            color = color,
            marker = calib_markers["Baseline"],
            linestyle=calib_styles["Baseline"],
            capsize=4,
            label=f"{angle2region[a]} Baseline"
        )
        plt.errorbar(
            E_4cal,
            med_4cal,
            yerr=[low_4cal, high_4cal],
            marker = calib_markers["4Cal"],
            color=color,
            linestyle = calib_styles["4Cal"],
            capsize=4,
            label=f"{angle2region[a]} Regional Scale"
        )
        
        plt.errorbar(
            E,
            med,
            yerr=[low, high],
            marker=calib_markers["4CalHitCal"],
            color = color,
            linestyle = calib_styles["4CalHitCal"],
            capsize=4,
            label=f"{angle2region[a]} Longitudinal Profile"
        )
        
        

    plt.xlabel("Energy [GeV]")
    plt.ylabel("Median Response")
    #plt.title(f"Median Response for {particle_labels[pid]} ")
    plt.ylim(0,1.2)
    plt.legend(frameon=False)
    
    plt.savefig(f"response_vs_energy_pid_{pid}_energys.pdf")
    plt.close()

for pid in particle:
    plt.figure(figsize=(7,5))

    for a in angles:
        E = np.array(results[pid][a]["energy"])
        med = np.array(results[pid][a]["median"])
        low = np.array(results[pid][a]["low"])
        high = np.array(results[pid][a]["high"])
        sigma = (high + low)/2
        resolution = sigma / med 
        E_base = np.array(results_base[pid][a]["energy"])
        med_base = np.array(results_base[pid][a]["median"])
        low_base = np.array(results_base[pid][a]["low"])
        high_base = np.array(results_base[pid][a]["high"])
        sigma_base = (high_base + low_base)/2
        resolution_base = sigma_base / med_base
        E_4cal = np.array(results_4cal[pid][a]["energy"])
        med_4cal = np.array(results_4cal[pid][a]["median"])
        low_4cal = np.array(results_4cal[pid][a]["low"])
        high_4cal = np.array(results_4cal[pid][a]["high"])
        sigma_4cal = (high_4cal + low_4cal)/2
        resolution_4cal = sigma_4cal / med_4cal
        region = angle2region[a]
        color = region_colors[region]
        #label=f"{angle2region[a]} 4Cal/HitCal"
        
        plt.plot(
            E_base,
            resolution_base,
            marker=calib_markers["Baseline"],
            color=color,
            linestyle = calib_styles["Baseline"],
            label=f"{angle2region[a]} Baseline"
        )
        plt.plot(
            E_4cal,
            resolution_4cal,
            marker=calib_markers["4Cal"],
            color=color,
            linestyle = calib_styles["4Cal"],
            label=f"{angle2region[a]} Regional Scale"
        )
        
        plt.plot(
            E, 
            resolution,
            marker=calib_markers["4CalHitCal"],
            color=color,
            linestyle = calib_styles["4CalHitCal"],
            label=f"{angle2region[a]} Longitudinal Profile"
        )
        

    plt.xlabel("Energy [GeV]")
    plt.ylabel(r"Resolution $\sigma/E$")
    #plt.title(f"Resolution for {particle_labels[pid]}")
    plt.ylim(0,1.0)
    plt.legend(frameon=False)
   
    plt.savefig(f"resolution_vs_energy_pid_{pid}_energys.pdf")
    plt.close()
'''