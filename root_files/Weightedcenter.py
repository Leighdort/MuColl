# Getting the energy weighted center

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec"
}

energies = [1, 2, 5, 10, 50, 100, 150, 200]

cluster_centers_per_energy = {}
for num in energies:

    print(f"\nProcessing {num} GeV...")
    file = uproot.open(f"reco_outpute{num}.edm4hep.root")
    events = file["events"]

    cluster_centers_per_energy[num] = []

    # Load cluster–hit mapping arrays
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()

    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

    # Preload calorimeter arrays
    pos = {}
    ener = {}

    for name in system2name.values():
        prefix = f"{name}/{name}"
        pos[name] = {
            "x": events[f"{prefix}.position.x"].array(),
            "y": events[f"{prefix}.position.y"].array(),
            "z": events[f"{prefix}.position.z"].array(),
        }
        ener[name] = events[f"{prefix}.energy"].array()

    # Loop over events
    for i in range(events.num_entries):

        if i % 100 == 0:
            print(f"  Event {i}")

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]

        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]

        event_centers = []

        # Loop over clusters in this event
        for j in range(len(hits_begin_arr)):

            start = hits_begin_arr[j]
            end   = hits_end_arr[j]

            # vectorized slices of hits
            indices = hit_index[start:end]
            ids     = collection_ID[start:end]

            if len(indices) == 0:
                continue

            # vectorized concatenation of hits across subsystems
            xs = np.empty(len(indices))
            ys = np.empty(len(indices))
            zs = np.empty(len(indices))
            ws = np.empty(len(indices))

            # group by system to avoid looping per hit
            for sysid in np.unique(ids):
                sysname = system2name[sysid]
                mask = (ids == sysid)
                idxs = indices[mask]

                xs[mask] = pos[sysname]["x"][i][idxs]
                ys[mask] = pos[sysname]["y"][i][idxs]
                zs[mask] = pos[sysname]["z"][i][idxs]
                ws[mask] = ener[sysname][i][idxs]

            # weighted centroid
            wsum = np.sum(ws)
            if wsum == 0:
                continue

            x_c = np.sum(xs * ws) / wsum
            y_c = np.sum(ys * ws) / wsum
            z_c = np.sum(zs * ws) / wsum

            event_centers.append((x_c, y_c, z_c))

        # store per event
        cluster_centers_per_energy[num].append(event_centers)

print("\nFinished computing cluster centers.\n")

# ----------------------------------------------------------
# Plot theta distribution for each energy
# ----------------------------------------------------------

for num in energies:

    all_centers = [c for evt in cluster_centers_per_energy[num] for c in evt]
    if len(all_centers) == 0:
        print(f"No clusters found for {num} GeV – skipping.")
        continue

    centers = np.array(all_centers)
    xs, ys, zs = centers.T

    theta = np.degrees(np.arctan2(np.sqrt(xs**2 + ys**2), zs))

    plt.figure()
    plt.hist(theta, bins=50, edgecolor="black")
    plt.title(f"Theta distribution for energy {num} GeV")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(f"theta_energy_{num}.png")
    plt.close()

print("All plots saved.")
