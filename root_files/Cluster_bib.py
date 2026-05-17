#Number of Clusters given 1 Original Clusters
#Here I am going to look at energy differences with one cluster
#Energy difference 1 cluster
#Energy difference between leading and secondary for bib when 1 cluster
#Submit_clusterbib.py

# Widthroot_fast.py
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
#WARNING
#SHOULD PRINT AS GOING TO REDUCE MEMORY AND TIME
system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [2, 10, 50]
choices = [2, 10, 50] #you don't need both, I just have both 
nclus_median = []
nclus_low = []
nclus_high = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    file_nobib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    events_nobib = file_nobib["events"]
    events_bib = file_bib["events"]
    pandora_clusters_nobib = events_nobib["PandoraClusters"]
    pandora_clusters_bib = events_bib["PandoraClusters"]
    nobib_energy = pandora_clusters_nobib["PandoraClusters.energy"].array()
    bib_energy = pandora_clusters_bib["PandoraClusters.energy"].array()
    num_clus = [] #number of bib clusters
    for i in range(events_nobib.num_entries):
        num_bib_clus = 0 #num bib clusters 
        if i % 1000 == 0:
            print(f"Event {i}")
        if len(nobib_energy[i]) == 1:
            num_bib_clus = len(bib_energy[i])
            num_clus.append(num_bib_clus)
    #Ok now after all events are accounted for
    num_clus = np.array(num_clus)
    #Repeat for number of clusters
    q16, q84 = np.percentile(num_clus, [16, 84])
    median = np.median(num_clus)
    nclus_low.append(median - q16)
    nclus_high.append(q84 - median)
    nclus_median.append(median)
    #I will likely have to adjust binning but this is temporary
    bins = np.arange(np.min(num_clus), np.max(num_clus) + 2) - 0.5
    plt.hist(num_clus, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Number of Bib Clusters given 1 Normal Cluster")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Number of Bib Clusters from 1 Nonbib Cluster {num} GeV Pions")
    plt.tight_layout()
    plt.savefig(f"nbibclus_pions_bib{num}GeV.pdf")
    plt.close()


#Third summary graph, number of bib clusters given 1 normal cluster
plt.errorbar(choices, nclus_median, yerr=[nclus_low, nclus_high], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Number of Bib Clusters")
plt.title("Number of Bib Clusters given original 1 Cluster Event")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_nbibclus_bib.pdf")
plt.close()

