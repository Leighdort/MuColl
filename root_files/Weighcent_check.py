#Checking Weightedcenter.py

#Is x, y, z in the clusters = energy weighted center?
#The hope is yes!

#Submit_weightcheck.sh
# Getting the energy weighted center
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
energies = [2]
#Let's just check no bibs, difference in theta and phi

for num in energies:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    events = file["events"]
    dif_x = []
    dif_y = []
    dif_z = []
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
    cluster_x=pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y=pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z=pandora_clusters["PandoraClusters.position.z"].array()

    # Preload calorimeter arrays
    pos = {}
    ener = {}

    for name in real_systems:
        prefix = f"{name}/{name}"
        pos[name] = {
            "x": events[f"{prefix}.position.x"].array(),
            "y": events[f"{prefix}.position.y"].array(),
            "z": events[f"{prefix}.position.z"].array(),
        }
        ener[name] = events[f"{prefix}.energy"].array()

    for i in range(events.num_entries):

        if i % 1000 == 0:
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
                if sysname is None or sysname == "Skip":
                    continue
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
            #These are the weighted centers
            clusx = cluster_x[i][j]
            clusy = cluster_y[i][j]
            clusz = cluster_z[i][j]
            dif_x.append(clusx-x_c)
            dif_y.append(clusy-y_c)
            dif_z.append(clusz-z_c)
    dif_x = np.array(dif_x)
    dif_y = np.array(dif_y)
    dif_z = np.array(dif_z)
    med_x = np.median(dif_x)
    med_y = np.median(dif_y)
    med_z = np.median(dif_z)
    std_x = np.std(dif_x, ddof=1)
    std_y = np.std(dif_y, ddof=1)
    std_z = np.std(dif_z, ddof=1)
    print("=== Difference statistics ===")
    print(f"X: median = {med_x:.4e}, std = {std_x:.4e}")
    print(f"Y: median = {med_y:.4e}, std = {std_y:.4e}")
    print(f"Z: median = {med_z:.4e}, std = {std_z:.4e}")
    
    plt.hist(dif_x, bins=10, edgecolor='black')
    plt.title("Difference in X")
    plt.xlabel("Difference")
    plt.tight_layout()
    plt.savefig("x_difference.png")
    plt.close()
    plt.hist(dif_y, bins=10, edgecolor='black')
    plt.title("Difference in Y")
    plt.xlabel("Difference")
    plt.tight_layout()
    plt.savefig("y_difference.png")
    plt.close()
    plt.hist(dif_z, bins=10, edgecolor='black')
    plt.title("Difference in Z")
    plt.xlabel("Difference")
    plt.tight_layout()
    plt.savefig("z_difference.png")
    plt.close()

print("All plots saved.")
