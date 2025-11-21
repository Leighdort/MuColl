#Widthroot.py
#You can only run this outside the apptainer
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import json
import uproot
#import ROOT

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec"
}

energies = [10, 50, 100, 150, 200]

sigma_mean_per_energy = []
sigma_std_per_energy = []
widths_per_energy = {}

for num in energies:
    file = uproot.open(f"reco_outpute{num}.edm4hep.root")
    events = file["events"]
    print(num)
    sigma_eta_list_energy = []
    widths_per_event = []

    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

    posx = {}
    posy = {}
    posz = {}
    energy = {}

    for name in system2name.values():
        prefix = f"{name}/{name}"
        posx[name]   = events[f"{prefix}.position.x"].array()
        posy[name]   = events[f"{prefix}.position.y"].array()
        posz[name]   = events[f"{prefix}.position.z"].array()
        energy[name] = events[f"{prefix}.energy"].array()
    
    for i in range(events.num_entries):
        if i % 100 == 0:
            print(f"Event {i}")
        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]

        hit_index = hit_index_all[i]
        collection_ID = collectionID_all[i]

        sigma_eta_percluster = []

        for j in range(len(hits_begin_arr)):

            hits_begin = hits_begin_arr[j]
            hits_end   = hits_end_arr[j]

            cluster_hit_indices = hit_index[hits_begin:hits_end]
            cluster_collection_ID = collection_ID[hits_begin:hits_end]

            xs, ys, zs, weights = [], [], [], []

            for k in range(len(cluster_hit_indices)):
                idx = cluster_hit_indices[k]
                sysname = system2name[cluster_collection_ID[k]]
              
                xs.append(posx[sysname][i][idx])
                ys.append(posy[sysname][i][idx])
                zs.append(posz[sysname][i][idx])
                weights.append(energy[sysname][i][idx])

            if len(xs) == 0:
                continue

            xs = np.array(xs)
            ys = np.array(ys)
            zs = np.array(zs)
            weights = np.array(weights)

            x_c = np.average(xs, weights=weights)
            y_c = np.average(ys, weights=weights)
            z_c = np.average(zs, weights=weights)

            r2 = (xs - x_c)**2 + (ys - y_c)**2
            r_rms = np.sqrt(np.average(r2, weights=weights))

            mag_c = np.sqrt(x_c**2 + y_c**2 + z_c**2)
            if mag_c == 0:
                continue

            theta_c = np.arccos(z_c / mag_c)
            eta_c = -np.log(np.tan(theta_c / 2))

            sigma_eta = np.arctan(r_rms / mag_c) * np.cosh(eta_c)
            sigma_eta_percluster.append(sigma_eta)

        if sigma_eta_percluster:
            mean_width = np.mean(sigma_eta_percluster)
            sigma_eta_list_energy.append(mean_width)
            widths_per_event.append((i, mean_width))

    if sigma_eta_list_energy:
        sigma_mean_per_energy.append(np.mean(sigma_eta_list_energy))
        sigma_std_per_energy.append(np.std(sigma_eta_list_energy))
        widths_per_energy[num] = widths_per_event
print("\n==== TOP 5 EVENTS PER ENERGY ====")
for E in energies:
    events_list = widths_per_energy[E]

    # sort high → low
    top5 = sorted(events_list, key=lambda x: x[1], reverse=True)[:5]

    print(f"\nEnergy {E} GeV — Top 5 widest events:")
    for idx, width in top5:
        print(f"  Event {idx:4d}   width = {width:.5f}")




print("==== RESULTS ====")
for E, widths in widths_per_energy.items():
    print(f"Energy {E} GeV:")
    print(f"  Mean width: {sigma_mean_per_energy[energies.index(E)]:.4f}")
    print(f"  Std width:  {sigma_std_per_energy[energies.index(E)]:.4f}")
    print(f"  Num clusters: {len(widths)}\n")


plt.errorbar(
    energies,
    sigma_mean_per_energy,
    yerr=sigma_std_per_energy,
    fmt='o-', capsize=5
)

plt.xlabel("Beam Energy (GeV)")
plt.ylabel(r"$\langle \sigma_\eta \rangle$   (mean cluster width)")
plt.title("Mean Cluster Width vs Energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_width_vs_energye.pdf")
plt.close()

for E in energies:
    plt.hist([w for (_, w) in widths_per_energy[E]], bins=30, alpha=0.7)
    plt.xlabel(r"$\sigma_\eta$ (width)")
    plt.ylabel("Cluster count")
    plt.title(f"Width distribution at {E} GeV")
    plt.tight_layout()
    plt.savefig(f"width_hist_{E}e.pdf")
    plt.close()


'''
data = {
    "energies": energies,
    "sigma_mean_per_energy": sigma_mean_per_energy,
    "sigma_std_per_energy": sigma_std_per_energy,
    "widths_per_energy": {str(E): widths_per_energy[E] for E in energies}
}

with open("cluster_widthse.json", "w") as f:
    json.dump(data, f, indent=4)
'''
