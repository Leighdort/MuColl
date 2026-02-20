#Barycenter_root_leading.py

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
    cluster_energy = pandora_clusters["PandoraClusters.energy"].array()
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
        cluster_energy_now = cluster_energy[i]
        index = np.argmax(cluster_energy_now)
        # --- Select leading cluster ---
        if len(cluster_energy_now) == 0:
            continue
        index = np.argmax(cluster_energy_now)
        start = hits_begin_arr[index]
        end   = hits_end_arr[index]
        if end <= start:
            continue
        indices = hit_index[start:end]
        ids     = collection_ID[start:end]
        # --- Collect hit positions and energies ---
        xs, ys, zs, ws = [], [], [], []
        for sysid in np.unique(ids):
            if sysid not in system2name:
                continue
            sysname = system2name[sysid]
            if sysname == "Skip":
                continue
            mask = ids == sysid
            idxs = indices[mask]
            xs.append(pos[sysname]["x"][i][idxs])
            ys.append(pos[sysname]["y"][i][idxs])
            zs.append(pos[sysname]["z"][i][idxs])
            ws.append(ener[sysname][i][idxs])
        # Concatenate
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        zs = np.concatenate(zs)
        ws = np.concatenate(ws)
        # --- Apply cuts ---
        finite = np.isfinite(ws)
        xs, ys, zs, ws = xs[finite], ys[finite], zs[finite], ws[finite]
        energy_cut = 1e-6
        valid = ws > energy_cut
        xs, ys, zs, ws = xs[valid], ys[valid], zs[valid], ws[valid]
        ws = np.asarray(ws)
        if ws.size == 0 or np.sum(ws) < 1e-3:
            continue
        # --- Barycenter ---
        x = np.average(xs, weights=ws)
        y = np.average(ys, weights=ws)
        z = np.average(zs, weights=ws)

        # --- Distance ---
        r = np.sqrt(x**2 + y**2 + z**2)
        energy_distance.append(r)

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
    cluster_energy = pandora_clusters["PandoraClusters.energy"].array()
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
        cluster_energy_now = cluster_energy[i]
        index = np.argmax(cluster_energy_now)
        # --- Select leading cluster ---
        if len(cluster_energy_now) == 0:
            continue
        index = np.argmax(cluster_energy_now)
        start = hits_begin_arr[index]
        end   = hits_end_arr[index]
        if end <= start:
            continue
        indices = hit_index[start:end]
        ids     = collection_ID[start:end]
        # --- Collect hit positions and energies ---
        xs, ys, zs, ws = [], [], [], []
        for sysid in np.unique(ids):
            if sysid not in system2name:
                continue
            sysname = system2name[sysid]
            if sysname == "Skip":
                continue
            mask = ids == sysid
            idxs = indices[mask]
            xs.append(pos[sysname]["x"][i][idxs])
            ys.append(pos[sysname]["y"][i][idxs])
            zs.append(pos[sysname]["z"][i][idxs])
            ws.append(ener[sysname][i][idxs])
        # Concatenate
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        zs = np.concatenate(zs)
        ws = np.concatenate(ws)
        # --- Apply cuts ---
        finite = np.isfinite(ws)
        xs, ys, zs, ws = xs[finite], ys[finite], zs[finite], ws[finite]
        energy_cut = 1e-6
        valid = ws > energy_cut
        xs, ys, zs, ws = xs[valid], ys[valid], zs[valid], ws[valid]
        ws = np.asarray(ws) 
        if ws.size == 0 or np.sum(ws) < 1e-3:
            continue
        # --- Barycenter ---
        x = np.average(xs, weights=ws)
        y = np.average(ys, weights=ws)
        z = np.average(zs, weights=ws)

        # --- Distance ---
        r = np.sqrt(x**2 + y**2 + z**2)
        energy_distance.append(r)

    energy_distance = np.array(energy_distance)
    pion_distance.append(np.median(energy_distance))
    q16, q84 = np.percentile(energy_distance, [16, 84])
    median = np.median(energy_distance)
    pion_distance_bottom.append(median - q16)
    pion_distance_top.append(q84 - median)
    bins = np.linspace(np.min(energy_distance), np.max(energy_distance), 30)
    plt.hist(energy_distance, bins=bins, edgecolor='black')
    plt.xlabel("Distance between cluster center to (0,0,0)")
    plt.axvline(median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}")
    plt.ylabel("Count")
    plt.title(f"Distance of Barycenter to (0,0,0) for {num} GeV Pions")
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
