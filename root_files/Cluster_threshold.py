import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

energies = [2, 10, 50]
max_clusters = 10
choices = [2, 10, 50]

cluster_counts = {num: np.zeros(max_clusters, dtype=int) for num in energies}

for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    events = file["events"]
    pandora_clusters = events["PandoraClusters"]
    cluster_energy = pandora_clusters["PandoraClusters.energy"].array()

    for i in range(events.num_entries):
        n = len(cluster_energy[i])
        if 1 <= n <= max_clusters:
            cluster_counts[num][n - 1] += 1

# Plot — energy on x axis, cumulative events on y axis
fig, ax = plt.subplots(figsize=(9, 6))

markers = ['o', 's', '^']
colors = ['blue', 'orange', 'red']

for cluster_n in range(max_clusters):
    # For each cluster count (1,2,3...) get cumulative sum up to that point for each energy
    y_vals = [np.sum(cluster_counts[num][:cluster_n + 1]) for num in energies]
    ax.plot(
        choices,
        y_vals,
        marker='o',
        label=f"Up to {cluster_n + 1} cluster(s)"
    )

ax.set_ylim(7000, 10500)
ax.set_xticks(choices)
ax.set_xlabel("Beam Energy (GeV)")
ax.set_ylabel("Cumulative Events")
ax.set_title("Cumulative Cluster Count by Energy (Pions)")
ax.legend(fontsize=7)
ax.grid(True)
plt.tight_layout()
plt.savefig("2stacked_clusters_pions.pdf")
plt.close()

print("\nDone!")

'''
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
pion_events = {
    2: [1568,7553,6837,9410,8006,8146,4010,9105,7811,7693,6790,7901,4635,9323,8916,7590,7684,8506,9779,5608],
    10: [7320,1630,54,1530,3183,3400,5896,4404,5841,1228,8502,3651,6668,7763,2920,4280,8706,4786,3480,9376],
    50: [7702,4875,2121,1817,9266,7122,8921,5783,9497,8726,4742,1244,4787,7103,8186,5361,6645,6766,3272,4434],
}
mc_median= []
mc_low = []
mc_high = []
fnobib_low=[]
fnobib_high=[]
fnobib_median=[]
fbib_low=[]
fbib_high=[]
fbib_median=[]
dbib_low=[]
dbib_high=[]
dbib_median=[]
dnobib_low=[]
dnobib_high=[]
dnobib_median=[]

one_cluster = []
two_cluster = []
three_cluster = []
four_cluster =[]
five_cluster = []
nobib_count = []
bib_count = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    fraction_bib = []
    fraction_nobib = []
    bib_distance = []
    nobib_distance = []
    fraction_events = []
    num_events_1_clus = 0
    passes = 0
    if num == 2:
        bound = 0.05
        #bound = .01
        #bound = 0.056
        #targets = [0.5, 1, 1.25, 2.5]
        #targets = [0.5, 1, 1.09, 1.75]
    if num == 10:
        bound = 0.02
        #targets = [0.5, 1, 1.15, 1.5]
        #targets = [0.5, 1, 1.2]
        #bound = .01
        #bound = 0.021
    if num == 50:
        bound = 0.006
        #targets = [0.8, 1.0, 1.11, 1.3]
        #bound = .01
        #targets = [0.2, 0.99, 1, 1.2]
        #bound = 0.0076
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    bib_events = file_bib["events"]
    events = file["events"]
    pandora_clusters_bib = bib_events["PandoraClusters"]
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
    cluster_x_bib=pandora_clusters_bib["PandoraClusters.position.x"].array()
    cluster_y_bib=pandora_clusters_bib["PandoraClusters.position.y"].array()
    cluster_z_bib=pandora_clusters_bib["PandoraClusters.position.z"].array()
    cluster_energy_bib=pandora_clusters_bib["PandoraClusters.energy"].array()
    one_cluster_cut = 0
    cluster_cut2 = 0
    cluster_cut3 = 0
    cluster_cut4 = 0
    cluster_cut5 = 0
    nobib_cut = 0
    bib_cut = 0 

    #Loop through just pions 
    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        
        if len(cluster_energy[i]) == 1:
            one_cluster_cut += 1 
        if len(cluster_energy[i]) == 2:
            cluster_cut2 +=1
        if len(cluster_energy[i]) == 3:
            cluster_cut3 +=1
        if len(cluster_energy[i]) ==  4:
            cluster_cut4 +=1
        if len(cluster_energy[i]) == 5:
            cluster_cut5 +=1

    one_cluster.append(one_cluster_cut)
    two_cluster.append(cluster_cut2)
    three_cluster.append(cluster_cut3)
    four_cluster.append(cluster_cut4)
    five_cluster.append(cluster_cut5)


one = np.array(one_cluster)
two = np.array(two_cluster)
three = np.array(three_cluster)
four = np.array(four_cluster)
five = np.array(five_cluster)

plt.stackplot(
    choices,
    one,
    two,
    three,
    four,
    five,
    labels=["1 cluster", "2 cluster", "3 cluster", "4 cluster", "5 cluster"]
)

plt.xlabel("Beam Energy")
plt.ylabel("Events")
plt.title("Num Cluster Cuts (Pions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("stacked_clusters_pions.pdf")
plt.close()

#divide nobib_count / one_cluster and plot that value

#Let's do a pion graph now
plt.scatter(choices, one_cluster, marker='o')
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the cut (10,000 Total)")
plt.title("Pion events that make the 1 Cluster Cut")
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_oneclus_pion.pdf")
plt.close()
'''