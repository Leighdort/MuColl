#Printing stats for bad
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
energies = [50]

sigma_mean_per_energy = []
sigma_std_per_energy = []
widths_per_energy = {}

#First we will do the electrons
#Now I will do the pions
print("this is actually the pion one not the other one")
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    start = 5900
    end   = 6000
    sigma_eta_list_energy = []
    widths_per_event = []

    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()[start:end]
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()[start:end]

    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()[start:end]
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()[start:end]

    # Pull all subsystem hit maps up front
    posx = {}
    posy = {}
    posz = {}
    energy_map = {}

    for name in real_systems:
        prefix = f"{name}/{name}"
        posx[name]   = events[f"{prefix}.position.x"].array()[start:end]
        posy[name]   = events[f"{prefix}.position.y"].array()[start:end]
        posz[name]   = events[f"{prefix}.position.z"].array()[start:end]
        energy_map[name] = events[f"{prefix}.energy"].array()[start:end]

    # Loop over events
    for i in range(end - start):

        if i % 1000 == 0:
            print(f"Event {i}")

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]

        hit_index = hit_index_all[i]
        collection_ID = collectionID_all[i]

        sigma_eta_percluster = []

        # Loop over clusters in the event
        for j in range(len(hits_begin_arr)):

            lo = hits_begin_arr[j]
            hi = hits_end_arr[j]

            idxs = hit_index[lo:hi]
            sysIDs = collection_ID[lo:hi]

            if len(idxs) == 0:
                continue

            sysnames = np.vectorize(system2name.get)(sysIDs)

            if 'None' in sysnames.astype(str):
                print(f"Skipping cluster {j} in event {i} at energy {num} GeV due to unknown system ID")
                continue
            
            mask = (sysnames != "Skip")
            sysnames = sysnames[mask]
            idxs = idxs[mask]
            if len(sysnames) == 0:
                continue

            # Build hit info
            xs = np.array([posx[s][i][idx] for s, idx in zip(sysnames, idxs)])
            ys = np.array([posy[s][i][idx] for s, idx in zip(sysnames, idxs)])
            zs = np.array([posz[s][i][idx] for s, idx in zip(sysnames, idxs)])
            weights = np.array([energy_map[s][i][idx] for s, idx in zip(sysnames, idxs)])

            # -----------------------------
            # 🔍 DEBUG: PRINT HIT DETAILS
            # -----------------------------
            print("\n--- DEBUG CLUSTER ---")
            print(f"Event:   {i}")
            print(f"Cluster: {j}")
            print(f"Num hits: {len(idxs)}")

            print("Energies:")
            #print(weights)

            #print("Positions (x,y,z):")
            #for xx, yy, zz in zip(xs, ys, zs):
            #    print(f"({xx:.2f}, {yy:.2f}, {zz:.2f})")
            #print("----------------------\n")

