#Distance between weighted center for more than one cluster

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

for num in energies:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    print(f"Processing {num} GeV")

    pandora_clusters = events["PandoraClusters"]
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    cluster_hit_begin = pandora_clusters["PandoraClusters.clusters_begin"].array()
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
    cluster_x = pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y = pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z = pandora_clusters["PandoraClusters.position.z"].array()
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()

    energy_distance = []
    total_num_clusters = 0
    big_num_clusters = 0

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

    for i in range(events.num_entries):
        total_num_clusters += 1
        if i % 100 == 0:
            print(f"  Event {i}")

        n_clusters = len(cluster_hit_begin[i])
        if n_clusters <= 1:
            continue
        big_num_clusters += 1

        #Making an array 
        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]

        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]

        # Calculate cluster centers (average of cluster positions)
        meanx = []
        meany = []
        meanz = []
        for j in range(n_clusters):
            start = hits_begin_arr[j]
            end   = hits_end_arr[j]
            # vectorized slices of hits
            indices = hit_index[start:end]
            ids     = collection_ID[start:end]

            if end - start == 0:  # skip empty clusters
                print("meow")
                continue
            xs = np.empty(len(indices))
            ys = np.empty(len(indices))
            zs = np.empty(len(indices))
            ws = np.empty(len(indices))

            #Doing the position array
            for sysid in np.unique(ids):
                if sysid not in system2name:
                    continue #I think skips the whole cluster 
                sysname = system2name[sysid]
                mask = (ids == sysid)
                idxs = indices[mask]
                xs[mask] = pos[sysname]["x"][i][idxs]
                ys[mask] = pos[sysname]["y"][i][idxs]
                zs[mask] = pos[sysname]["z"][i][idxs]
                ws[mask] = ener[sysname][i][idxs]
            onemean_x = np.average(xs, weights=ws)
            onemean_y = np.average(ys, weights=ws)
            onemean_z = np.average(zs, weights=ws)
            meanx.append(onemean_x)
            meany.append(onemean_y)
            meanz.append(onemean_z)
        w = 0
        distance = []
        n_valid = len(meanx)
        while w < n_valid:
            x = meanx[w]
            y = meany[w]
            z = meanz[w]
            q = w + 1
            while q <n_valid:
                x2 = meanx[q]
                y2 = meany[q]
                z2 = meanz[q]
                r = np.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)
                distance.append(r)
                q += 1
            w += 1
        if distance:
            distance_mean = np.mean(distance)
            energy_distance.append(distance_mean)
    energy_distance = np.array(energy_distance)
    bins = np.linspace(0, np.max(energy_distance), 30)
    plt.hist(energy_distance, bins=bins, edgecolor='black')
    plt.xlabel("Distance between cluster centers")
    plt.ylabel("Count")
    plt.title(f"Distance when more than one cluster for {num} GeV Electrons")
    plt.tight_layout()
    plt.savefig(f"cluster_distance_electrons10x{num}GeV.pdf")
    plt.close()
    print(total_num_clusters)
    print(big_num_clusters)


