#Width 

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

energies = [10, 50, 100, 150, 200]
sigma_mean_per_energy = []
sigma_std_per_energy = []
widths_per_energy = {E: [] for E in energies}

#Ok now we are going to try to get rid of particles more than 0.5 away, has more than 10% of the energy
removed = {}
for energy in energies:
    df = pd.read_csv(f"outputpion_reco{energy}_hits.csv")
    kept_events = []
    remove_number = 0
    num_removed = 0
    print(energy)
    # Loop over events
    for event_id, event_group in df.groupby("event"):
        x1_theta, x2_phi, x3_layer, clusters = [], [], [], []
        print(event_id)
        # Convert (x, y, z) → (theta, phi, layer)
        for _, row in event_group.iterrows():
            x, y, z = row["x"], row["y"], row["z"]
            r = math.sqrt(x * x + y * y + z * z)
            cluster_num, layer, system = row["cluster_num"], row["layer"], row["system"]

            theta = math.acos(z / r)
            phi = math.atan2(y, x)
            if phi < 0:
                phi += 2 * math.pi

            # Offset layers for HCAL systems
            if system in ["hcal endcap", "hcal barrel"]:
                x3_layer.append(layer + 50)
            else:
                x3_layer.append(layer)

            x1_theta.append(theta)
            x2_phi.append(phi)
            clusters.append(cluster_num)

        # Build DataFrame for this event
        df_plot = pd.DataFrame({
            "theta": x1_theta,
            "phi": x2_phi,
            "layer": x3_layer,
            "cluster_num": clusters,
            "energy": event_group["energy"].values,
            "event": event_id,
            "x": event_group["x"].values,
            "y": event_group["y"].values,
            "z": event_group["z"].values,
            "system": event_group["system"].values,
            "layer_raw": event_group["layer"].values
        })
        event_passes = True
        # Loop over clusters in this event
        for cluster_id, groups in df_plot.groupby("cluster_num"):
            points = groups[["theta", "phi", "layer"]].values
            weights = groups["energy"].values
            print(cluster_id)
            # Energy-weighted center
            center = np.average(points, axis=0, weights=weights)

            # Weight and center points
            points_centered = points - center
            points_weighted = points_centered * np.sqrt(weights)[:, None]

            # SVD → principal axis
            _, _, vh = np.linalg.svd(points_weighted, full_matrices=False)
            direction = vh[0]
            D_unit = direction / np.linalg.norm(direction)

            # Compute perpendicular distances
            distances = np.linalg.norm(
                points - center - np.dot(points - center, D_unit)[:, None] * D_unit,
                axis=1
            )

            total_cluster_energy = weights.sum()
            far_energy_fraction = weights[distances > 0.5].sum() / total_cluster_energy
            if far_energy_fraction > 0.10:
                event_passes = False
                num_removed += 1
                break
        if event_passes:
            kept_events.append(df_plot)
    if kept_events:
        final_df = pd.concat(kept_events, ignore_index = True)
        final_df.to_csv(f"filtered_reco{energy}_hits.csv", index=False)
    else:
        print(f"No events kept for energy {energy}")
    removed[energy] = num_removed
    remove_number=remove_number + 1
print(removed)




for energy in energies:
    #df = pd.read_csv(f"outputelectron_reco{energy}_hits.csv")
    df = pd.read_csv(f"filtered_reco{energy}_hits.csv")
    sigma_eta_list_energy = []

    for event_id, event_group in df.groupby("event"):
        sigma_eta_percluster = []

        for cluster_id, cluster_group in event_group.groupby("cluster_num"):
            x_points = cluster_group["x"].values
            y_points = cluster_group["y"].values
            z_points = cluster_group["z"].values
            weights_xyz = cluster_group["energy"].values

            # Cluster center
            x_c = np.average(x_points, weights=weights_xyz)
            y_c = np.average(y_points, weights=weights_xyz)
            z_c = np.average(z_points, weights=weights_xyz)

            # Transverse RMS
            r2 = (x_points - x_c)**2 + (y_points - y_c)**2
            r_rms = np.sqrt(np.average(r2, weights=weights_xyz))

            # Pseudorapidity of cluster centroid
            mag_c = np.sqrt(x_c**2 + y_c**2 + z_c**2)
            theta_c = np.arccos(z_c / mag_c)
            eta_c = -np.log(np.tan(theta_c / 2))

            # ATLAS-style width
            sigma_eta = np.arctan(r_rms / mag_c) * np.cosh(eta_c)
            sigma_eta_percluster.append(sigma_eta)

        if sigma_eta_percluster:
            sigma_eta_list_energy.append(np.mean(sigma_eta_percluster))

    if sigma_eta_list_energy:
        sigma_mean_per_energy.append(np.mean(sigma_eta_list_energy))
        sigma_std_per_energy.append(np.std(sigma_eta_list_energy))
        widths_per_energy[energy] = sigma_eta_list_energy

    plt.hist(widths_per_energy[energy], bins=30, alpha=0.7)
    plt.xlabel(r"$\sigma_{\eta}$ (width)")
    plt.ylabel("Cluster count")
    plt.title(f"Width distribution at {energy}")
    plt.tight_layout()
    plt.savefig(f"pionremovehistogramwidth{energy}.pdf")
    plt.close()




#Looking at singular event, energy 100 event 10

'''
energies = [100]
sigma_mean_per_energy = []
sigma_std_per_energy = []
widths_per_energy = {E: [] for E in energies}

for energy in energies:
    df = pd.read_csv(f"outputpion_reco{energy}_hits.csv")
    sigma_eta_list_energy = []
    mask = df["event"] == 10
    df=df[mask]
    for event_id, event_group in df.groupby("event"):
        sigma_eta_percluster = []

        for cluster_id, cluster_group in event_group.groupby("cluster_num"):
            x_points = cluster_group["x"].values
            y_points = cluster_group["y"].values
            z_points = cluster_group["z"].values
            weights_xyz = cluster_group["energy"].values

            # Cluster center
            x_c = np.average(x_points, weights=weights_xyz)
            y_c = np.average(y_points, weights=weights_xyz)
            z_c = np.average(z_points, weights=weights_xyz)

            # Transverse RMS
            r2 = (x_points - x_c)**2 + (y_points - y_c)**2
            r_rms = np.sqrt(np.average(r2, weights=weights_xyz))

            # Pseudorapidity of cluster centroid
            mag_c = np.sqrt(x_c**2 + y_c**2 + z_c**2)
            theta_c = np.arccos(z_c / mag_c)
            eta_c = -np.log(np.tan(theta_c / 2))

            # ATLAS-style width
            sigma_eta = np.arctan(r_rms / mag_c) * np.cosh(eta_c)
            sigma_eta_percluster.append(sigma_eta)

        if sigma_eta_percluster:
            sigma_eta_list_energy.append(np.mean(sigma_eta_percluster))

    if sigma_eta_list_energy:
        sigma_mean_per_energy.append(np.mean(sigma_eta_list_energy))
        sigma_std_per_energy.append(np.std(sigma_eta_list_energy))
        widths_per_energy[energy] = sigma_eta_list_energy


plt.hist(widths_per_energy[100], bins=30, alpha=0.7)
plt.xlabel(r"$\sigma_{\eta}$ (width)")
plt.ylabel("Cluster count")
plt.title("Width distribution at 100 GeV, event 10")
plt.tight_layout()
plt.savefig(f"pionhistogramwidth100.10.pdf")
plt.close()
'''
