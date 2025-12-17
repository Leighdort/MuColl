#Numclustersroot.py
#Here I will find the average number of clusters per energy 
#I will look at number of clusters per each event then bin that per energy then graph
#I will then find delta r between clusters

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot


#First we will get mean and stdev
#And we will graph log scale
#Then we will pick some events, deal
energies = [1, 2, 5, 10, 50, 100, 150, 200]

#Now we resume what we need
electron_mean = []
electron_low = []
electron_high = []
pion_mean = []
pion_low = []
pion_high = []

for num in energies:
    num_clusters_array = []
    thetas = []
    theta2 = []
    theta3 = []
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    print(num)
    pandora_clusters = events["PandoraClusters"]
    cluster_hit_begin = pandora_clusters["PandoraClusters.clusters_begin"].array()
    cluster_x = pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y = pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z = pandora_clusters["PandoraClusters.position.z"].array()
    for i in range(events.num_entries):
        num_clusters_array.append(len(cluster_hit_begin[i]))
        for j in range(len(cluster_x[i])):
            x = cluster_x[i][j]
            y = cluster_y[i][j]
            z = cluster_z[i][j]
            r = np.sqrt(x*x + y*y + z*z)
            theta = np.arccos(z / r)
            theta_deg = np.degrees(theta)
            thetas.append(theta_deg)
    
    thetas_array = np.array(thetas)
    binst = np.linspace(0, 360, 362)
    num_clusters_array = np.array(num_clusters_array)
    electron_mean.append(np.median(num_clusters_array))
    median_val = np.median(num_clusters_array)
    q16, q84 = np.percentile(num_clusters_array, [16, 84]) 
    electron_low.append(median_val - q16)
    electron_high.append(q84 - median_val)
    bins = np.arange(0, num_clusters_array.max()+2)
    plt.hist(num_clusters_array, bins=bins, edgecolor='black')
    plt.yscale("log")
    plt.axvline(median_val,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median_val:.2f}")
    plt.legend()
    plt.xlabel("Number of Clusters per event")
    plt.ylabel("Count")
    plt.title(f"Clusters for {num} GeV Electrons")
    plt.tight_layout()
    plt.savefig(f"clusterselectrons10x_{num}GeV.pdf")
    plt.close()
    
    #plt.hist(thetas_array, bins=binst, edgecolor='black')
    #plt.xlabel("Theta Values")
    #plt.ylabel("Count")
    #plt.title(f"Thetas for {num} GeV Electrons")
    #plt.tight_layout()
    #plt.savefig(f"centerelectrons10x_{num}GeV.pdf")
    #plt.close()
    

 
for num in energies:
    num_clusters_array = []
    thetas = []
    theta2 = []
    theta3 = []
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    print(num)
    pandora_clusters = events["PandoraClusters"]
    cluster_hit_begin = pandora_clusters["PandoraClusters.clusters_begin"].array()
    cluster_x = pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y = pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z = pandora_clusters["PandoraClusters.position.z"].array()
    for i in range(events.num_entries):
        num_clusters_array.append(len(cluster_hit_begin[i]))
        for j in range(len(cluster_x[i])):
            x = cluster_x[i][j]
            y = cluster_y[i][j]
            z = cluster_z[i][j]
            r = np.sqrt(x*x + y*y + z*z)
            theta = np.arccos(z / r)
            theta_deg = np.degrees(theta)
            thetas.append(theta_deg)
    
    thetas_array = np.array(thetas)
    binst = np.linspace(0, 360, 362)
    num_clusters_array = np.array(num_clusters_array)
    pion_mean.append(np.median(num_clusters_array))
    median_val = np.median(num_clusters_array)
    q16, q84 = np.percentile(num_clusters_array, [16, 84]) 
    pion_low.append(median_val - q16)
    pion_high.append(q84 - median_val)
    bins = np.arange(0, num_clusters_array.max()+2)
    plt.hist(num_clusters_array, bins=bins, edgecolor='black')
    plt.yscale("log")
    plt.axvline(median_val,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median_val:.2f}")
    plt.xlabel("Number of clusters per event")
    plt.ylabel("Count")
    plt.legend()
    plt.title(f"Clusters for {num} GeV Pions")
    plt.tight_layout()
    plt.savefig(f"clusterspions10x_{num}GeV.pdf")
    plt.close()

plt.errorbar(energies, pion_mean, yerr=[pion_low, pion_high], fmt='s', capsize=4, alpha=0.6, label="Pions")
plt.errorbar(energies, electron_mean, yerr=[electron_low, electron_high], fmt='o', capsize=4, alpha=0.6, label="Electrons")
plt.xlabel("Beam Energy")
plt.ylabel("Median number of Clusters per Event")
plt.title("Median cluster multiplicity vs energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_cluster_count.pdf")
plt.close()


'''
electron_lengths = {
    1: [1,5],
    2: [1,6],
    5: [1,5],
    10: [1,6],
    50: [1,6],
    100: [1,5,19],
    150: [1,6,17],
    200: [1,5,7]
}

pion_lengths = {
    1: [0,1,7],
    2: [1,11],
    5: [1,22],
    10: [1,27],
    50: [1,39],
    100: [1,48],
    150: [1,2,48],
    200: [1,2,54]
}


energies = [1, 2, 5, 10, 50, 100, 150, 200]


# Electrons
for num in energies:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    pandora_clusters = events["PandoraClusters"]
    cluster_hit_begin = pandora_clusters["PandoraClusters.clusters_begin"].array()
    
    lengths_to_find = electron_lengths.get(num, [])
    found = set()
    
    for i in range(events.num_entries):
        n_clusters = len(cluster_hit_begin[i])
        if n_clusters in lengths_to_find and n_clusters not in found:
            print(f"Electron example: {num} GeV, Event {i}, Clusters: {n_clusters}")
            found.add(n_clusters)
        if len(found) == len(lengths_to_find):
            break  # Stop after finding one example per desired length

# Pions
for num in energies:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    pandora_clusters = events["PandoraClusters"]
    cluster_hit_begin = pandora_clusters["PandoraClusters.clusters_begin"].array()
    
    lengths_to_find = pion_lengths.get(num, [])
    found = set()
    
    for i in range(events.num_entries):
        n_clusters = len(cluster_hit_begin[i])
        if n_clusters in lengths_to_find and n_clusters not in found:
            print(f"Pion example: {num} GeV, Event {i}, Clusters: {n_clusters}")
            found.add(n_clusters)
        if len(found) == len(lengths_to_find):
            break  # Stop after finding one example per desired length

'''