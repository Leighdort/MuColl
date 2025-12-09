#Checking weird counts

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
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec", "EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
weird = []
normal = []
clusters_total= []
polluted_total = []
energies = [1, 2, 5, 10, 50, 100, 150, 200]
for num in energies:
    count_weird = 0
    count_total = 0
    clusters = 0
    polluted = 0
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    # Load cluster–hit mapping arrays
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
    # Preload calorimeter arrays
    pos = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        pos[name] = {
            "x": events[f"{prefix}.position.x"].array(),
            "y": events[f"{prefix}.position.y"].array(),
            "z": events[f"{prefix}.position.z"].array(),
        }

    # Loop over events
    for i in range(events.num_entries):
        if i % 100 == 0:
            print(f"  Event {i}")

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]
        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]
        # Loop over clusters in this event
        for j in range(len(hits_begin_arr)):
            polute_boolean = False
            clusters +=1
            start = hits_begin_arr[j]
            end = hits_end_arr[j]
            # vectorized slices of hits
            indices = hit_index[start:end]
            ids = collection_ID[start:end]
            count_total +=len(ids)
            if len(indices) == 0:
                continue
            for sysid in ids:
                sysname = system2name[sysid]
                if sysname == "Skip":
                    count_weird +=1
                    polute_boolean = True
            if polute_boolean == True:
                polluted +=1
    clusters_total.append(clusters)
    polluted_total.append(polluted)
    weird.append(count_weird)
    normal.append(count_total)
weird = np.array(weird)
plt.figure(figsize=(10,5))
plt.plot(energies, weird, marker='o')
plt.xlabel("Energy")
plt.ylabel("Weird_hits Count")
plt.title("Electrons Weird Counts")
plt.grid(True)
plt.savefig(f"weirdhits_electrons.pdf")
plt.close()
normal = np.array(normal)
ratio = np.array(weird/normal)
polluted_total = np.array(polluted_total)
clusters_total = np.array(clusters_total)
ratio_clusters = np.array(polluted_total/clusters_total)
plt.figure(figsize=(10,5))
plt.plot(energies, ratio, marker='o')
plt.xlabel("Energy")
plt.ylabel("Weird hits Ratio")
plt.title("Electrons Weird/Normal Ratio")
plt.grid(True)
plt.savefig(f"weird_normalhits_electrons.pdf")
plt.close()
plt.figure(figsize=(10,5))
plt.plot(energies, polluted_total, marker='o')
plt.xlabel("Energy")
plt.ylabel("Weird ClusterCount")
plt.title("Electrons Weird Counts")
plt.grid(True)
plt.savefig(f"weird_cluster_electrons.pdf")
plt.close()
plt.plot(energies, ratio_clusters, marker='o')
plt.xlabel("Energy")
plt.ylabel("Weird Cluster/ Normal Ratio")
plt.title("Electrons Weird/Normal Ratio")
plt.grid(True)
plt.savefig(f"weird_normal_cluster_electrons.pdf")
plt.close()

