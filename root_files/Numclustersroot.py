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

energies = [1, 2, 5, 10, 50, 100, 150, 200]

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
    bins = np.arange(0, num_clusters_array.max()+2)
    plt.hist(num_clusters_array, bins=bins, edgecolor='black')
    plt.xlabel("Number of clusters per event")
    plt.ylabel("Count")
    plt.title(f"Clusters for {num} GeV Electrons")
    plt.tight_layout()
    plt.savefig(f"clusterselectrons10x_{num}GeV.pdf")
    plt.close()

    plt.hist(thetas_array, bins=binst, edgecolor='black')
    plt.xlabel("Theta Values")
    plt.ylabel("Count")
    plt.title(f"Thetas for {num} GeV Electrons")
    plt.tight_layout()
    plt.savefig(f"centerelectrons10x_{num}GeV.pdf")
    plt.close()

 
