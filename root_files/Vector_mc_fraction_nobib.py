#Playing around checking Lorentz 4 Vector


import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
import vector



#We want to use vector supposedly

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

mc_median= []
mc_low = []
mc_high = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    fraction = []
    num_events_1_clus = 0
    passes = 0
    if num == 2:
        bound = 0.05
    if num == 10:
        bound = 0.02
    if num == 50:
        bound = 0.06
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
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
    mc_momx = mcparticles["MCParticles.momentum.x"].array()
    mc_momy = mcparticles["MCParticles.momentum.y"].array()
    mc_momz = mcparticles["MCParticles.momentum.z"].array()
    mc_mass = mcparticles["MCParticles.mass"].array()
    #Now I want cluster energy
    #Also I want cluster theta, phi
    cluster_x=pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y=pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z=pandora_clusters["PandoraClusters.position.z"].array()
    cluster_energy=pandora_clusters["PandoraClusters.energy"].array()
    angular_dist = []
    regular_dist = []
    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i]) == 1:
            num_events_1_clus +=1 
            mask = (status_mc[i] == 1)
            mx=mc_x[i][mask]
            my=mc_y[i][mask]
            mz=mc_z[i][mask]
            momx=mc_momx[i][mask]
            momy=mc_momy[i][mask]
            momz=mc_momz[i][mask]
            mcmass=mc_mass[i][mask]
            cx = cluster_x[i][0] #mind you, only works for 1 cluster
            cy = cluster_y[i][0]
            cz = cluster_z[i][0]
            #I am now going to vectorize and use three vectors
            mc_arr = np.array([mx[0], my[0], mz[0]])
            cl_arr = np.array([cx, cy, cz])
            cos_angle = np.dot(mc_arr, cl_arr) / (np.linalg.norm(mc_arr) * np.linalg.norm(cl_arr))
            cos_angle = np.clip(cos_angle, -1, 1)  # avoid floating point errors
            angular_distance = np.arccos(cos_angle)
            if angular_distance <= bound:
                passes += 1
                mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                fraction_mc = cluster_energy[i][0] / mc_energy[0]
                fraction.append(fraction_mc)

    print(f"Number of 1 Event Clusters: {num_events_1_clus}")
    print(f"Number of Events that fall within margin: {passes}")
    fraction = np.array(fraction)
    q16, q84 = np.percentile(fraction, [16, 84])
    median = np.median(fraction)
    mc_low.append(median-q16)
    mc_high.append(q84-median)
    mc_median.append(median)
    bins=30
    plt.hist(fraction, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Cluster/MC Energy Fraction")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Fraction given 1 Cluster within Bounds Pion Nobib (Vector) {num}")
    plt.tight_layout()
    plt.savefig(f"dif_mc_fraction_nobib{num}GeV.pdf")
    plt.close()

#Third summary graph, number of bib clusters given 1 normal cluster
plt.errorbar(choices, mc_median, yerr=[mc_low, mc_high], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Cluster/MC Energy")
plt.title("Median Cluster/MC Energy Given 1 Nobib Cluster within Bounds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dif_summary_mc_nobib.pdf")
plt.close()


#mcfractionbib_
