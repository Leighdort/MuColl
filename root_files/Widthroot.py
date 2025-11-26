# Widthroot_fast.py
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
    2381985645: "HcalEndcapCollectionRec"
}

energies = [10, 50, 100, 150, 200]

sigma_mean_per_energy = []
sigma_std_per_energy = []
widths_per_energy = {}

for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    file = uproot.open(f"reco_outpute{num}.edm4hep.root")
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

    for name in system2name.values():
        prefix = f"{name}/{name}"
        posx[name]   = events[f"{prefix}.position.x"].array()
        posy[name]   = events[f"{prefix}.position.y"].array()
        posz[name]   = events[f"{prefix}.position.z"].array()
        energy_map[name] = events[f"{prefix}.energy"].array()

    # Loop over events
    for i in range(events.num_entries):

        if i % 100 == 0:
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

# ---- PRINT RESULTS ----

print("\n==== TOP 5 EVENTS PER ENERGY ====")
for E in energies:
    events_list = widths_per_energy[E]
    top5 = sorted(events_list, key=lambda x: x[1], reverse=True)[:5]
    print(f"\nEnergy {E} GeV — Top 5 widest events:")
    for idx, width in top5:
        print(f"  Event {idx:4d}   width = {width:.5f}")

print("\n==== SUMMARY ====")
for E in energies:
    i = energies.index(E)
    print(f"Energy {E} GeV:")
    print(f"  Mean width: {sigma_mean_per_energy[i]:.4f}")
    print(f"  Std width:  {sigma_std_per_energy[i]:.4f}")
    print(f"  Num events: {len(widths_per_energy[E])}")

# ---- PLOTTING ----

plt.errorbar(
    energies,
    sigma_mean_per_energy,
    yerr=sigma_std_per_energy,
    fmt='o-', capsize=5
)
plt.xlabel("Beam Energy (GeV)")
plt.ylabel(r"$\langle \sigma_\eta \rangle$")
plt.title("Mean Cluster Width vs Energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_width_vs_energy.pdf")
plt.close()

for E in energies:
    plt.hist([w for (_, w) in widths_per_energy[E]], bins=30, alpha=0.7)
    plt.xlabel(r"$\sigma_\eta$")
    plt.ylabel("Count")
    plt.title(f"Cluster Widths at {E} GeV")
    plt.tight_layout()
    plt.savefig(f"width_hist_{E}.pdf")
    plt.close()
