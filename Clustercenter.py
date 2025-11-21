import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

energies = [10, 50, 100, 150, 200]
average_distance_per_energy = []
average_std_per_energy = []
sigma_mean_per_energy =[]
sigma_std_per_energy = []

#This is finding the mean + 2 * stdev
max10 = 0.09908603012684407 + 2*0.2496409450330537
max50 = 0.10385793958957643 + 2*0.2802210659824645
max100 = 0.09687807361857798 + 2*0.2663190622414585
max150 = 0.09278983525668952 + 2*0.24678349436908809
max200 = 0.09497690456891297 + 2*0.26645795525693133

max10sigma =0.1245968389625622 + 2*0.04326702688880244
max50sigma = 0.1338012716356742 + 2* 0.038992616731815595
max100sigma = 0.1344906925863698 + 2*0.03707671423197496
max150sigma = 0.13623111055079948 + 2*0.036300301739066294
max200sigma = 0.13542127303737075 + 2*0.0359903281870181

max10eventssigma = []
max50eventssigma = []
max100eventssigma = []
max150eventssigma = []
max200eventssigma = []


#These are the max events & Distances
max10events = []
max10eventsdist = []
max50events = []
max50eventsdist = []
max100events = []
max100eventsdist = []
max150events = []
max150eventsdist = []
max200events = []
max200eventsdist = []
#I am storing now average distance per event
energy10 = []
energy50 = []
energy100 = []
energy150 = []
energy200 = []


for energy in energies:
    df = pd.read_csv(f"outputelectron_reco{energy}_hits.csv")

    average_distance_all_events = []  # one average per event
    sigma_eta_list_energy = []
    # Loop over events
    for event_id, event_group in df.groupby("event"):
        x1_theta, x2_phi, x3_layer, clusters = [], [], [], []

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
            "cluster": clusters,
            "energy": event_group["energy"].values
        })

        # Loop over clusters in this event
        cluster_mean_distance = {}
        for cluster_id, group in df_plot.groupby("cluster"):
            points = group[["theta", "phi", "layer"]].values
            weights = group["energy"].values

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

            #Now we must communite weighted distance
            weighted_avg_distance = np.average(distances, weights=weights)
            cluster_mean_distance[cluster_id] = weighted_avg_distance
        # Event-level average across clusters
        if len(cluster_mean_distance) > 0:
            avg_cluster_distance = np.mean(list(cluster_mean_distance.values()))
            average_distance_all_events.append(avg_cluster_distance)
        #We must commute width now
        sigma_eta_percluster = []
        for cluster_id, group in df_plot.groupby("cluster"):
            #We are going to recalculate everything
            # --- Calculate cluster center in x, y, z ---
            cluster_mask = event_group["cluster_num"].values == cluster_id
            x_points = event_group.loc[cluster_mask, "x"].values
            y_points = event_group.loc[cluster_mask, "y"].values
            z_points = event_group.loc[cluster_mask, "z"].values
            weights_xyz = event_group.loc[cluster_mask, "energy"].values  # per-point weights

            # Now np.average works because shapes match
            x_c = np.average(x_points, weights=weights_xyz)
            y_c = np.average(y_points, weights=weights_xyz)
            z_c = np.average(z_points, weights=weights_xyz)

            cluster_center_xyz = np.array([x_c, y_c, z_c])

            # Transverse RMS distance
            r2 = (x_points - x_c)**2 + (y_points - y_c)**2
            r_rms = np.sqrt(np.average(r2, weights=weights_xyz))

            # Pseudorapidity of cluster centroid
            mag_c = np.sqrt(x_c**2 + y_c**2 + z_c**2)
            theta_c = np.arccos(z_c / mag_c)
            eta_c = -np.log(np.tan(theta_c / 2))

            # ATLAS-style cluster width
            sigma_eta = np.arctan(r_rms / mag_c) * np.cosh(eta_c)
            sigma_eta_percluster.append(sigma_eta)
        sigma_eta_per_event = np.mean(sigma_eta_percluster)
        sigma_eta_list_energy.append(sigma_eta_per_event)
    if sigma_eta_list_energy:
        sigma_mean_per_energy.append(np.mean(sigma_eta_list_energy))
        sigma_std_per_energy.append(np.std(sigma_eta_list_energy))
        #sigma eta is my width

#    #Finding the "outliers", and printing their energy
#    for i, distance in enumerate(average_distance_all_events):
#        if energy == 10: 
#            energy10.append(distance)
#            if distance > max10:
#                max10events.append(i)
#                max10eventsdist.append(distance)
#        elif energy == 50:
#            energy50.append(distance)
#            if distance > max50:
#                max50events.append(i)
#                max50eventsdist.append(distance)
#        elif energy == 100:
#            energy100.append(distance)
#            if distance > max100:
#                max100events.append(i)
#                max100eventsdist.append(distance)
#        elif energy == 150:
#            energy150.append(distance)
#           if distance > max150:
#                max150events.append(i)
#               max150eventsdist.append(distance)
#        elif energy == 200:
#            energy200.append(distance)
#            if distance > max200:
#                max200events.append(i)
#                max200eventsdist.append(distance)
#    # Mean across all events at this energy
    for i, sigma in enumerate(sigma_eta_list_energy) :
        if energy == 10:
            if sigma > max10sigma:
                max10eventssigma.append(i)
        if energy == 50:
            if sigma > max50sigma:
                max50eventssigma.append(i)
        if energy == 100:
            if sigma > max100sigma:
                max100eventssigma.append(i)
        if energy == 150:
            if sigma > max150sigma:
                max150eventssigma.append(i)
        if energy == 200:
            if sigma > max200sigma:
                max200eventssigma.append(i)
    if len(average_distance_all_events) > 0:
        overall_mean = np.mean(average_distance_all_events)
        overall_std = np.std(average_distance_all_events)
        average_distance_per_energy.append(overall_mean)
        average_std_per_energy.append(overall_std)
    else:
        average_distance_per_energy.append(np.nan)

print("Average cluster distance per energy:")
print(average_distance_per_energy)
print(average_std_per_energy)
print(sigma_mean_per_energy)
print(sigma_std_per_energy)

print("Event 10")
print(max10eventssigma)
print("Event 50")
print(max50eventssigma)
print("Event 100")
print(max100eventssigma)
print("Event 150")
print(max150eventssigma)
print("Event 200")
print(max200eventssigma)
########################################################################
results = {}
for energy in energies:
    # Read cluster centers
    df_clusters = pd.read_csv(f"outputelectron_reco{energy}_clusters.csv")
    results[energy] = {}
    for event_id, event_group in df_clusters.groupby("event"):
        results[energy][event_id] = {}
        for cluster_id, cluster_group in event_group.groupby("cluster_num"):
            results[energy][event_id][cluster_id] = {
                "theta": cluster_group["theta"].values[0],
                "phi": cluster_group["phi"].values[0]
            }

# Store the average distance per energy
average_per_energy = {}

for energy in energies:
    df_hits = pd.read_csv(f"outputelectron_reco{energy}_hits.csv")
    event_means = []  # mean per event for this energy

    for event_id, event_group in df_hits.groupby("event"):
        cluster_means = []  # mean per cluster in this event

        for cluster_id, group in event_group.groupby("cluster_num"):
            points = group[["x", "y", "z"]].values
            weights = group["energy"].values

            # Convert to theta and phi
            x, y, z = group["x"].values, group["y"].values, group["z"].values
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

            # Get cluster center
            if (energy in results and
                event_id in results[energy] and
                cluster_id in results[energy][event_id]):

                cluster_theta = results[energy][event_id][cluster_id]["theta"]
                cluster_phi = results[energy][event_id][cluster_id]["phi"]

                # Compute angular difference
                dtheta = theta - cluster_theta
                dphi = np.mod(phi - cluster_phi + np.pi, 2 * np.pi) - np.pi
                angular_distance = np.sqrt(dtheta**2 + dphi**2)

                # Weighted mean per cluster
                weighted_mean = np.average(angular_distance, weights=weights)
                cluster_means.append(weighted_mean)

        # Mean over all clusters in this event
        if len(cluster_means) > 0:
            event_mean = np.mean(cluster_means)
            event_means.append(event_mean)

    # Mean over all events in this energy
    if len(event_means) > 0:
        average_per_energy[energy] = np.mean(event_means)
    else:
        average_per_energy[energy] = np.nan

print(average_per_energy)

###############################################
#These graphs are from above:
#print("Average cluster distance per energy:")
#print(average_distance_per_energy)
#print(average_std_per_energy)
#print("E10")
#print(max10events)
#print("distances")
#print(max10eventsdist)
#print("E50")
#print(max50events)
#print("distances")
#print(max50eventsdist)
#print("E100")
#print(max100events)
#print("distances")
#print(max100eventsdist)
#print("E150")
#print(max150events)
#print("distances")
#print(max150eventsdist)
#print("E200")
#print(max200events)
#print("distances")
#print(max200eventsdist)

#Create a graph
#plt.figure(figsize=(7,5))
#plt.hist(energy10, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#plt.axvline(max10, color='red', linestyle='--', linewidth = 2, label="Mean + 2 sigma")
#plt.title("Distribution of Mean Cluster Distances (10 GeV)")
#plt.xlabel("Mean Distance per Event")
#plt.ylabel("Number of Events")
#plt.legend()
# Save as PDF
#plt.tight_layout()
#plt.savefig("energy10_histogram.pdf")

#Create a graph
#plt.figure(figsize=(7,5))
#plt.hist(energy50, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#plt.axvline(max50, color='red', linestyle='--', linewidth = 2, label="Mean + 2 sigma")
#plt.title("Distribution of Mean Cluster Distances (50 GeV)")
#plt.xlabel("Mean Distance per Event")
#plt.ylabel("Number of Events")
#plt.legend()
# Save as PDF
#plt.tight_layout()
#plt.savefig("energy50_histogram.pdf")

#Create a graph
#plt.figure(figsize=(7,5))
#plt.hist(energy100, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#plt.axvline(max100, color='red', linestyle='--', linewidth = 2, label="Mean + 2 sigma")
#plt.title("Distribution of Mean Cluster Distances (100 GeV)")
#plt.xlabel("Mean Distance per Event")
#plt.ylabel("Number of Events")
#plt.legend()
# Save as PDF
#plt.tight_layout()
#plt.savefig("energy100_histogram.pdf")

#Create a graph
#plt.figure(figsize=(7,5))
#plt.hist(energy150, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#plt.axvline(max150, color='red', linestyle='--', linewidth = 2, label="Mean + 2 sigma")
#plt.title("Distribution of Mean Cluster Distances (150 GeV)")
#plt.xlabel("Mean Distance per Event")
#plt.ylabel("Number of Events")
#plt.legend()
# Save as PDF
#plt.tight_layout()
#plt.savefig("energy150_histogram.pdf")

#Create a graph
#plt.figure(figsize=(7,5))
#plt.hist(energy200, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
#plt.axvline(max200, color='red', linestyle='--', linewidth = 2, label="Mean + 2 sigma")
#plt.title("Distribution of Mean Cluster Distances (10 GeV)")
#plt.xlabel("Mean Distance per Event")
#plt.ylabel("Number of Events")
#plt.legend()
# Save as PDF
#plt.tight_layout()
#plt.savefig("energy200_histogram.pdf")