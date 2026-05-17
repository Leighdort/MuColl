
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

#Changing to R and theta

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
#Then change it to R, phi, and theta 
#Then do this and positrons & make summary
#Then do with just everything + leading cluster


energies = [1, 2, 5, 10, 50, 100, 150, 200]
electron_mean = []
electron_low = []
electron_high = []
pion_mean = []
pion_low = []
pion_high = []
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    if not np.isfinite(r) or r < 1e-6:
        return None, None, None
    costheta = z / r
    costheta = np.clip(costheta, -1.0, 1.0)  # supposedly this is because sometimes costheta in python can be > or < this, which is not mathematically true
    theta = np.arccos(costheta)
    phi = np.arctan2(y, x)   
    return r, theta, phi
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
        mean_r = []
        mean_theta = []
        mean_phi = []
        for j in range(n_clusters):
            print("Unique system IDs in this event:", np.unique(sysIDs))
            start = hits_begin_arr[j]
            end   = hits_end_arr[j]
            # vectorized slices of hits
            indices = hit_index[start:end]
            ids     = collection_ID[start:end]

            if end - start == 0:  # skip empty clusters
                continue
            xs = np.zeros(len(indices)) #don't do np.empty!!!
            ys = np.zeros(len(indices))
            zs = np.zeros(len(indices))
            ws = np.zeros(len(indices)) 
            #We're skipping the punch through 
            valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
            if not np.any(valid_mask):
                continue  # skip cluster if all hits are "Skip"
            #Doing the position array
            for sysid in np.unique(ids[valid_mask]):
                if sysid not in system2name:
                    continue #I think skips the whole cluster 
                sysname = system2name[sysid]
                mask = (ids == sysid) & valid_mask
                idxs = indices[mask]
                xs[mask] = pos[sysname]["x"][i][idxs]
                ys[mask] = pos[sysname]["y"][i][idxs]
                zs[mask] = pos[sysname]["z"][i][idxs]
                ws[mask] = ener[sysname][i][idxs]
            #Ok this is gettign the weighted mean in Cartesian
            #Now we have to do an energy protection here
            energy_cut = 1e-6
            valid_energy = ws > energy_cut
            xs = xs[valid_energy]
            ys = ys[valid_energy]
            zs = zs[valid_energy]
            ws = ws[valid_energy]
            if ws.size == 0 or np.sum(ws) < 1e-3:
                continue
            onemean_x = np.average(xs, weights=ws)
            onemean_y = np.average(ys, weights=ws)
            onemean_z = np.average(zs, weights=ws)
            #Now we are going to convert to spherical
            r, theta, phi = cartesian_to_spherical(onemean_x, onemean_y, onemean_z)
            if r is None:
                continue
            if not (np.isfinite(r) and np.isfinite(theta) and np.isfinite(phi)):
                continue
            mean_r.append(r) #these things hold our center values for r, theta, phi
            mean_theta.append(theta)
            mean_phi.append(phi)
        if len(mean_r) <= 1:
            continue
        w = 0
        distance = []
        n_valid = len(mean_r)
        while w < n_valid:
            r1 = mean_r[w]
            theta1 = mean_theta[w]
            phi1 = mean_phi[w]
            q = w + 1
            while q <n_valid:
                r2 = mean_r[q]
                theta2 = mean_theta[q]
                phi2 = mean_phi[q]
                d = np.sqrt(r1**2 + r2**2 - 2*r1*r2*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2) + np.cos(theta1)*np.cos(theta2)))
                distance.append(d)
                q += 1
            w += 1
        if distance:
            distance_mean = np.mean(distance) #This is taking the average distance between each cluster, in R 
            energy_distance.append(distance_mean)
    energy_distance = np.array(energy_distance)
    median = np.median(energy_distance)
    q16, q84 = np.percentile(energy_distance, [16, 84])
    electron_mean.append(median)
    electron_low.append(q16)
    electron_high.append(q84)
    bins = np.linspace(0, np.max(energy_distance), 30)
    plt.hist(energy_distance, bins=bins, edgecolor='black')
    plt.xlabel("Distance between cluster centers")
    plt.ylabel("Count")
    plt.title(f"Average Spherical Distance between all Cluster Centers {num} GeV Electrons")
    plt.tight_layout()
    plt.savefig(f"cluster_distance_electrons10x{num}GeV.pdf")
    plt.close()
    #print(total_num_clusters)
    #print(big_num_clusters)


#Now we are going to do Pions


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
        if n_clusters <= 1:
            continue
        big_num_clusters += 1

        #Making an array 
        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]

        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]

        # Calculate cluster centers (average of cluster positions)
        mean_r = []
        mean_theta = []
        mean_phi = []
        for j in range(n_clusters):
            start = hits_begin_arr[j]
            end   = hits_end_arr[j]
            # vectorized slices of hits
            indices = hit_index[start:end]
            ids     = collection_ID[start:end]

            if end - start == 0:  # skip empty clusters
                continue
            xs = np.zeros(len(indices))
            ys = np.zeros(len(indices))
            zs = np.zeros(len(indices))
            ws = np.zeros(len(indices)) 
            #We're skipping the punch through 
            valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
            if not np.any(valid_mask):
                continue  # skip cluster if all hits are "Skip"
            #Doing the position array
            for sysid in np.unique(ids[valid_mask]):
                if sysid not in system2name:
                    continue #I think skips the whole cluster 
                sysname = system2name[sysid]
                mask = (ids == sysid) & valid_mask
                idxs = indices[mask]
                xs[mask] = pos[sysname]["x"][i][idxs]
                ys[mask] = pos[sysname]["y"][i][idxs]
                zs[mask] = pos[sysname]["z"][i][idxs]
                ws[mask] = ener[sysname][i][idxs]
            energy_cut = 1e-6
            valid_energy = ws > energy_cut
            xs = xs[valid_energy]
            ys = ys[valid_energy]
            zs = zs[valid_energy]
            ws = ws[valid_energy]
            if ws.size == 0 or np.sum(ws) < 1e-3:
                continue
            #Ok this is gettign the weighted mean in Cartesian
            onemean_x = np.average(xs, weights=ws)
            onemean_y = np.average(ys, weights=ws)
            onemean_z = np.average(zs, weights=ws)
            #Now we are going to convert to spherical
            r, theta, phi = cartesian_to_spherical(onemean_x, onemean_y, onemean_z)
            if r is None:
                continue
            if not (np.isfinite(r) and np.isfinite(theta) and np.isfinite(phi)):
                continue
            mean_r.append(r) #these things hold our center values for r, theta, phi
            mean_theta.append(theta)
            mean_phi.append(phi)
        if len(mean_r) <= 1:
            continue
        w = 0
        distance = []
        n_valid = len(mean_r)
        while w < n_valid:
            r1 = mean_r[w]
            theta1 = mean_theta[w]
            phi1 = mean_phi[w]
            q = w + 1
            while q <n_valid:
                r2 = mean_r[q]
                theta2 = mean_theta[q]
                phi2 = mean_phi[q]
                d = np.sqrt(r1**2 + r2**2 - 2*r1*r2*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2) + np.cos(theta1)*np.cos(theta2)))
                distance.append(d)
                q += 1
            w += 1
        if distance:
            distance_mean = np.mean(distance) #This is taking the average distance between each cluster, in R 
            energy_distance.append(distance_mean)
    energy_distance = np.array(energy_distance)
    median = np.median(energy_distance)
    q16, q84 = np.percentile(energy_distance, [16, 84])
    pion_mean.append(median)
    pion_low.append(q16)
    pion_high.append(q84)
    bins = np.linspace(0, np.max(energy_distance), 30)
    plt.hist(energy_distance, bins=bins, edgecolor='black')
    plt.xlabel("Distance between cluster centers")
    plt.ylabel("Count")
    plt.title(f"Average Spherical Distance between all Cluster Centers {num} GeV Pions")
    plt.tight_layout()
    plt.savefig(f"cluster_distance_pions10x{num}GeV.pdf")
    plt.close()
    #print(total_num_clusters)
    #print(big_num_clusters)

#Ok now we are going to do our summery graphs
plt.errorbar(energies, electron_mean, yerr=[electron_low, electron_high], alpha=0.6, fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_mean, yerr=[pion_low, pion_high], alpha = 0.6, fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy")
plt.ylabel("Median distance between Cluster Centers")
plt.title("Median distance between all Cluster Centers in an Event")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_cluster_r_all.pdf")
plt.close()


