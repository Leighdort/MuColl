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
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]


energies = [1, 2, 5, 10, 50, 100, 150, 200]
#First we're going to do electrons
elec_distance = []
elec_distance_bottom = []
elec_distance_top = []
pion_distance = []
pion_distance_bottom = []
pion_distance_top = []

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
    for name in real_systems:
        prefix = f"{name}/{name}"
        pos[name] = {
            "x": events[f"{prefix}.position.x"].array(),
            "y": events[f"{prefix}.position.y"].array(),
            "z": events[f"{prefix}.position.z"].array(),
        }
        ener[name] = events[f"{prefix}.energy"].array()

    for i in range(events.num_entries):
        total_num_clusters += 1
        if i % 1000 == 0:
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
            xs = np.zeros(len(indices))
            ys = np.zeros(len(indices))
            zs = np.zeros(len(indices))
            ws = np.zeros(len(indices))

            valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
            if not np.any(valid_mask):
                continue  # skip cluster if all hits are "Skip"
            #Doing the position array
            for sysid in np.unique(ids[valid_mask]):
                if sysid not in system2name:
                    print(f"Unknown id: {sysid}, from cluster {j}, from event {i}, with energy {num}")
                    continue
                sysname = system2name[sysid]
                mask = (ids == sysid) & valid_mask
                idxs = indices[mask]
                xs[mask] = pos[sysname]["x"][i][idxs]
                ys[mask] = pos[sysname]["y"][i][idxs]
                zs[mask] = pos[sysname]["z"][i][idxs]
                ws[mask] = ener[sysname][i][idxs]
        
            finite_mask = np.isfinite(ws)
            xs = xs[finite_mask]
            ys = ys[finite_mask]
            zs = zs[finite_mask]
            ws = ws[finite_mask]
            energy_cut = 1e-6 #energy cut specific particle
            valid_energy = ws > energy_cut
            xs = xs[valid_energy]
            ys = ys[valid_energy]
            zs = zs[valid_energy]
            ws = ws[valid_energy]
            if ws.size == 0 or np.sum(ws) < 1e-3: #If cluster itself is small 
                print("meow")
                continue
                
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
    elec_distance.append(np.median(energy_distance))
    q16, q84 = np.percentile(energy_distance, [16, 84])
    median = np.median(energy_distance)
    elec_distance_bottom.append(median - q16)
    elec_distance_top.append(q84 - median)
    bins = np.linspace(np.min(energy_distance), np.max(energy_distance), 30)
    plt.hist(energy_distance, bins=bins, edgecolor='black')
    plt.xlabel("Distance between cluster center to (0,0,0)")
    plt.axvline(median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}")
    plt.ylabel("Count")
    plt.title(f"Distance of Barycenter to (0,0,0) for {num} GeV Electrons")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"cluster_barycenter_electrons_{num}GeV.pdf")
    plt.close()

for num in energies:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
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
    for name in real_systems:
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
            xs = np.zeros(len(indices))
            ys = np.zeros(len(indices))
            zs = np.zeros(len(indices))
            ws = np.zeros(len(indices))

            valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
            if not np.any(valid_mask):
                continue  # skip cluster if all hits are "Skip"
            #Doing the position array
            for sysid in np.unique(ids[valid_mask]):
                if sysid not in system2name:
                    print(f"Unknown id: {sysid}, from cluster {j}, from event {i}, with energy {num}")
                    continue
                sysname = system2name[sysid]
                mask = (ids == sysid) & valid_mask
                idxs = indices[mask]
                xs[mask] = pos[sysname]["x"][i][idxs]
                ys[mask] = pos[sysname]["y"][i][idxs]
                zs[mask] = pos[sysname]["z"][i][idxs]
                ws[mask] = ener[sysname][i][idxs]

            finite_mask = np.isfinite(ws)
            xs = xs[finite_mask]
            ys = ys[finite_mask]
            zs = zs[finite_mask]
            ws = ws[finite_mask]
            energy_cut = 1e-6 #energy cut specific particle
            valid_energy = ws > energy_cut
            xs = xs[valid_energy]
            ys = ys[valid_energy]
            zs = zs[valid_energy]
            ws = ws[valid_energy]
            if ws.size == 0 or np.sum(ws) < 1e-3: #If cluster itself is small 
                print("meow")
                continue
                
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
    pion_distance.append(np.median(energy_distance))
    q16, q84 = np.percentile(energy_distance, [16, 84])
    median = np.median(energy_distance)
    pion_distance_bottom.append(median - q16)
    pion_distance_top.append(q84 - median)
    bins = np.linspace(np.min(energy_distance), np.max(energy_distance), 30)
    plt.hist(energy_distance, bins=bins, edgecolor='black')
    plt.xlabel("Distance between cluster center to (0,0,0)")
    plt.ylabel("Count")
    plt.title(f"Distance of Barycenter to (0,0,0) for {num} GeV Pions")
    plt.axvline(median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"cluster_barycenter_pions_{num}GeV.pdf")
    plt.close()


plt.errorbar(energies, elec_distance, yerr=[elec_distance_bottom, elec_distance_top], fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_distance, yerr=[pion_distance_bottom, pion_distance_top], fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy")
plt.ylabel("Average Barycenter")
plt.title("Average Barycenter per energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_barycenter.pdf")
plt.close()

