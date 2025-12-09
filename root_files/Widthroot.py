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
#energies = [1, 2, 5, 10, 50, 100, 150, 200]
#energies = [1, 2, 5]
#energies = [10, 50, 100]
#energies = [150]
energies = [200]

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

    sigma_eta_list_energy = []
    widths_per_event = []

    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()

    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

    # Pull all subsystem hit maps up front
    posx = {}
    posy = {}
    posz = {}
    energy_map = {}

    for name in real_systems:
        prefix = f"{name}/{name}"
        posx[name]   = events[f"{prefix}.position.x"].array()
        posy[name]   = events[f"{prefix}.position.y"].array()
        posz[name]   = events[f"{prefix}.position.z"].array()
        energy_map[name] = events[f"{prefix}.energy"].array()

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
            lo = hits_begin_arr[j]
            hi = hits_end_arr[j]

            idxs = hit_index[lo:hi]
            sysIDs = collection_ID[lo:hi]

            if len(idxs) == 0:
                continue

            # Vectorized system name lookups
            sysnames = np.vectorize(system2name.get)(sysIDs)

            if 'None' in sysnames.astype(str):
                print(f"Skipping cluster {j} in event {i} at energy {num} GeV due to unknown system ID")
                continue
            
            mask = (sysnames != "Skip")
            sysnames = sysnames[mask]
            idxs = idxs[mask]
            if len(sysnames) == 0:
                continue
            # Build arrays of hit info in one pass
            xs = np.array([posx[s][i][idx] for s, idx in zip(sysnames, idxs)])
            ys = np.array([posy[s][i][idx] for s, idx in zip(sysnames, idxs)])
            zs = np.array([posz[s][i][idx] for s, idx in zip(sysnames, idxs)])
            weights = np.array([energy_map[s][i][idx] for s, idx in zip(sysnames, idxs)])

            if weights.sum() == 0:
                continue

            # Weighted centroid (vectorized)
            x_c = np.sum(xs * weights) / np.sum(weights)
            y_c = np.sum(ys * weights) / np.sum(weights)
            z_c = np.sum(zs * weights) / np.sum(weights)

            # RMS in (x,y)
            r2 = (xs - x_c)**2 + (ys - y_c)**2
            r_rms = np.sqrt(np.average(r2, weights=weights))

            # Convert to eta-space
            mag_c = np.sqrt(x_c**2 + y_c**2 + z_c**2)
            if mag_c == 0:
                continue

            theta_c = np.arccos(z_c / mag_c)
            eta_c = -np.log(np.tan(theta_c / 2.0))

            sigma_eta = np.arctan(r_rms / mag_c) * np.cosh(eta_c)
            sigma_eta_percluster.append(sigma_eta)

        # End cluster loop

        if sigma_eta_percluster:
            mean_width = np.mean(sigma_eta_percluster)
            sigma_eta_list_energy.append(mean_width)
            widths_per_event.append((i, mean_width))

    # End event loop

    if sigma_eta_list_energy:
        sigma_mean_per_energy.append(np.mean(sigma_eta_list_energy))
        sigma_std_per_energy.append(np.std(sigma_eta_list_energy))
        widths_per_energy[num] = widths_per_event
        plt.hist([w for (_, w) in widths_per_energy[num]], bins=30, alpha=0.7)
        plt.xlabel(r"$\sigma_\eta$")
        plt.ylabel("Count")
        plt.title(f"Cluster Widths at {num} GeV")
        plt.tight_layout()
        plt.savefig(f"width_hist_{num}_pions10x.pdf")
        plt.close()
        avg_width = sigma_mean = sigma_mean_per_energy[-1]
        avg_std = sigma_std_per_energy[-1]
        print(f"Energy {num} GeV → Mean width = {avg_width:.5f}, Std = {avg_std:.5f}")
