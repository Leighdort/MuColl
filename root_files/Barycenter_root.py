#Barycenter_root.py

#The trick for this one is just finding where the clusters start?
#Maybe we will look at distance from 0,0,0

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



energies = [10, 50, 100, 150, 200]

for num in energies:
    file = uproot.open(f"reco_outpute{num}.edm4hep.root")
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
        n_valid = len(meanx)
        w = 0
        while w < n_valid:
            x = meanx[w]
            y = meany[w]
            z = meanz[w]
            r = np.sqrt((x)**2 + (y)**2 + (z)**2)
            energy_distance.append(r)
            w = w+1
    energy_distance = np.array(energy_distance)
    bins = np.linspace(np.min(energy_distance), np.max(energy_distance), 30)
    plt.hist(energy_distance, bins=bins, edgecolor='black')
    plt.xlabel("Distance between cluster center to (0,0,0)")
    plt.ylabel("Count")
    plt.title(f"Distance of Barycenter to (0,0,0) for {num} GeV Electrons")
    plt.tight_layout()
    plt.savefig(f"cluster_barycenter_{num}GeV.pdf")
    plt.close()

