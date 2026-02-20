#Distance to truth

#This is assuming cluster position is the energy weighted center
#---------> I believe this is correct

#What’s the average non-bib distance from the cluster to mc particle (looking at weighted cluster center) ** angular distance **
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [2, 10, 50]
choices = [2, 10, 50]
nclus_median = []
nclus_low = []
nclus_high = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    #file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    events = file["events"]
    pandora_clusters = events["PandoraClusters"]
    #I need the theta phi of mc and of clusters 
    mcparticles = events["MCParticles"]
    #we want a status of 1 
    status_mc = mcparticles["MCParticles.generatorStatus"].array()
    #do I want vertex or endpoint? I presume endpoint and we will see both
    #I will try it first with endpoint
    mc_x = mcparticles["MCParticles.endpoint.x"].array()
    mc_y = mcparticles["MCParticles.endpoint.y"].array()
    mc_z = mcparticles["MCParticles.endpoint.z"].array()
    #Now I want cluster energy
    #Also I want cluster theta, phi
    cluster_x=pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y=pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z=pandora_clusters["PandoraClusters.position.z"].array()
    cluster_energy=pandora_clusters["PandoraClusters.energy"].array()

    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i] == 1):
            mask = (status_mx[i] == 1)
            mx=mc_x[i][mask]
            my=mc_y[i][mask]
            mz=mc_z[i][mask]
            mc_r = np.sqrt(mx**2 + my**2 + mz**2)
            mc_theta = np.arccos(mz, mc_r) #these may all be in radians
            mc_phi = np.arctan2(my, mz)
            cx = cluster_x[i][0] #mind you, only works for 1 cluster
            cy = cluster_y[i][0]
            cz = cluster_z[i][0]
            c_r = np.sqrt(cx**2 + cy**2 + cz**2)
            c_theta = np.arccos(cz, c_r)
            c_phi = np.arccos(cy, cz)
            distance = np.sqrt(mc_r**2 + c_r**2 * (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi-c_phi) + np.cos(c_theta)*np.cos(mc_theta)))
            
            #Continue now adding
            #But first make sure x, y, z are the energy weighted centers using Weightecenter.py





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