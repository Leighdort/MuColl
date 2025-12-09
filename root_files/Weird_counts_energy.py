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
    3403901740: "MuonEndcapHits",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec", "EcalEndcapCollectionRec", "HcalEndcapCollectionRec", "MuonEndcapHits"]
average_energy = []
energy_std = []
average_ratio = []
ratio_std = []
type_counts_all_energies = {}
energies = [1, 2, 5, 10, 50, 100, 150, 200]
print("Pions")
for num in energies:
    type_per_energy = []
    total_energy_cluster = []
    missed_energy = []
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    # Load cluster–hit mapping arrays
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
    cluster_energy = pandora_clusters["PandoraClusters.energy"].array()
    muon = events["MuonEndcapHits"]
    muonenergy = muon["MuonEndcapHits.energy"].array()
    types = muon["MuonEndcapHits.type"].array()
    # Preload calorimeter arrays

    # Loop over events
# Loop over events
    for i in range(events.num_entries):
        if i % 1000 == 0:
            print(f"Event {i}")
        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]
        hit_index = hit_index_all[i]
        collection_ID = collectionID_all[i]
        sigma_eta_percluster = []
        # Loop over clusters in the event
        for j in range(len(hits_begin_arr)):
            # cluster = slice of hits
            total_energy = cluster_energy[i][j]
            lo = hits_begin_arr[j]
            hi = hits_end_arr[j]
            idxs = hit_index[lo:hi]
            sysIDs = collection_ID[lo:hi]
            if len(idxs) == 0:
                continue
            sysnames = np.vectorize(system2name.get)(sysIDs)
            mask = (sysnames == "MuonEndcapHits")
            muon_idxs = idxs[mask]
            if len(muon_idxs) !=0:
                particle_energy = 0
                for particle in muon_idxs:
                    particle_energy+=muonenergy[i][particle]
                    type_per_energy.append(types[i][particle])
                missed_energy.append(particle_energy)
                total_energy_cluster.append(total_energy)


    missed_energy = np.array(missed_energy)
    total_energy_cluster = np.array(total_energy_cluster)
    type_per_energy = np.array(type_per_energy)

    # Avoid division warnings
    mask_nonzero = total_energy_cluster > 0
    ratio = missed_energy[mask_nonzero] / total_energy_cluster[mask_nonzero]

    # -------- Plot 1: Histogram of missed energy --------
    if len(missed_energy):
        plt.figure(figsize=(10,5))
        plt.hist(missed_energy, bins=30)
        plt.xlabel("Missed Muon Energy")
        plt.ylabel("Clusters")
        plt.title(f"Histogram: Missed Energy at {num} GeV Pions")
        plt.grid(True)
        plt.savefig(f"missed_energy_hist_{num}elecGeV.pdf")
        plt.close()

    # -------- Plot 2: Histogram of missed/total ratio --------
    if len(ratio) > 0:
        plt.figure(figsize=(10,5))
        plt.hist(ratio, bins=30)
        plt.xlabel("Missed / Total Cluster Energy")
        plt.ylabel("Clusters")
        plt.title(f"Histogram: Missed/Total Energy Ratio at {num} GeV Pions")
        plt.grid(True)
        plt.savefig(f"missed_total_ratio_hist_elec{num}GeV.pdf")
        plt.close()
    
    #I want to do a histogram of total particles of each type
    if len(type_per_energy) > 0:
        plt.figure(figsize=(10,5))
        plt.hist(
            type_per_energy,
            bins=np.arange(type_per_energy.min() - 0.5,
                        type_per_energy.max() + 1.5, 1)
        )
        #Histogram of Hit types
        plt.xlabel("MuonEndcapHits Type")
        plt.ylabel("Count")
        plt.title(f"Histogram: Muon Hit Types at {num} GeV Pions")
        plt.grid(True)
        plt.savefig(f"muon_types_hist_elec{num}GeV.pdf")
        plt.close()
    
    #Saving things for the full file
    if len(missed_energy) > 0:
        average_energy.append(np.mean(missed_energy))
        energy_std.append(np.std(missed_energy))
    else:
        average_energy.append(0)
        energy_std.append(0)
    if len(ratio) > 0:
        average_ratio.append(np.mean(ratio))
        ratio_std.append(np.std(ratio))
    else:
        average_ratio.append(0)
        ratio_std.append(0)

    type_counts_all_energies[num] = {
        t: np.count_nonzero(type_per_energy == t)
        for t in np.unique(type_per_energy)}

#Printing Ratio 
print(average_energy)
print(energy_std)
print(average_ratio)
print(ratio_std)
print("\n=== PARTICLE TYPE COUNTS PER ENERGY ===")
for E in energies:
    print(f"\nEnergy = {E} GeV")
    for t, c in sorted(type_counts_all_energies[E].items()):
        print(f"  Type {t}: {c}")

