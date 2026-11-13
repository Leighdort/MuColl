#For 

#Bib response and resolution
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

angle = [15, 85]
particle = [211]
energy = [10, 30, 50]
geom_colors = {
    15: "tab:blue",     # Endcap
    85: "tab:orange"    # Barrel
}

particle_labels = {
    11: "Electron",
    211: "Pion"
}

line_styles = {
    "baseline": "-",
    "4cal": "--",
    "4calhitcal": ":"
}

markers = {
    "baseline": "o",
    "4cal": "s",
    "4calhitcal": "^"
}

for pid in particle:
    results = {
    "baseline_median": {15: [], 85: []},
    "baseline_low": {15: [], 85: []},
    "baseline_high": {15: [], 85: []},
    "baseline_resolution": {15: [], 85: []},

    "4cal_median": {15: [], 85: []},
    "4cal_low": {15: [], 85: []},
    "4cal_high": {15: [], 85: []},
    "4cal_resolution": {15: [], 85: []},

    "4calhitcal_median": {15: [], 85: []},
    "4calhitcal_low": {15: [], 85: []},
    "4calhitcal_high": {15: [], 85: []},
    "4calhitcal_resolution": {15: [], 85: []},
    }
    for a in angle:
        for num in energy:
            if (num == 50):
                file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_software_on.root")
                tree = file["MatchedClusters"]
                arrays = tree.arrays(library="np")
                matched_energy = arrays["matched_energy"]
                mc_energy = arrays["mc_energy"]
                mc_energy = np.asarray(mc_energy).reshape(-1)
                matched_energy = np.asarray(matched_energy).reshape(-1)
                response = matched_energy / mc_energy
                q16, q84 = np.percentile(response, [16, 84])
                median = np.median(response)
                sigma = (q84 - q16)/2
                resolution = sigma / median
                results["baseline_median"][a].append(median)
                results["baseline_low"][a].append(median-q16)
                results["baseline_high"][a].append(q84-median)
                results["baseline_resolution"][a].append(resolution)

                file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_calib10.root")
                tree = file["MatchedClusters"]
                arrays = tree.arrays(library="np")
                matched_energy = arrays["matched_energy"]
                mc_energy = arrays["mc_energy"]
                mc_energy = np.asarray(mc_energy).reshape(-1)
                matched_energy = np.asarray(matched_energy).reshape(-1)
                response = matched_energy / mc_energy
                q16, q84 = np.percentile(response, [16, 84])
                median = np.median(response)
                sigma = (q84 - q16)/2
                resolution = sigma / median
                results["4cal_median"][a].append(median)
                results["4cal_low"][a].append(median-q16)
                results["4cal_high"][a].append(q84-median)
                results["4cal_resolution"][a].append(resolution)

                if (pid == 211):
                    ecal_barrel = arrays["ecal_barrel"]
                    hcal_barrel = arrays["hcal_barrel"]
                    ecal_endcap = arrays["ecal_endcap"]
                    hcal_endcap = arrays["hcal_endcap"]
                    ecal_barrel = np.asarray(ecal_barrel).reshape(-1)
                    hcal_barrel = np.asarray(hcal_barrel).reshape(-1)
                    ecal_endcap = np.asarray(ecal_endcap).reshape(-1)
                    hcal_endcap = np.asarray(hcal_endcap).reshape(-1)
                    ecal_ratio = ecal_barrel + ecal_endcap
                    hcal_ratio = hcal_barrel + hcal_endcap
                    total = ecal_ratio + hcal_ratio
                    ecal_ratio = ecal_ratio / total
                    hcal_ratio = hcal_ratio / total

                    for i in range(len(response)):
                        baseline = response[i]
                        correction_1 = (ecal_endcap[i] / total[i]) * .09
                        correction_2 =  (hcal_endcap[i] / total[i]) * .03 
                        correction_3 = (ecal_barrel[i] / total[i]) * .16
                        correction_4 =  (hcal_barrel[i]  / total[i]) * .19
                        new_fraction = baseline - correction_1 + correction_2 - correction_3 + correction_4
                        response[i] = new_fraction

                    q16, q84 = np.percentile(response, [16, 84])
                    median = np.median(response)
                    sigma = (q84 - q16)/2
                    resolution = sigma / median
                    results["4calhitcal_median"][a].append(median)
                    results["4calhitcal_low"][a].append(median-q16)
                    results["4calhitcal_high"][a].append(q84-median)
                    results["4calhitcal_resolution"][a].append(resolution)
            else:
                file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_uncalibrated_base.root")
                tree = file["MatchedClusters"]
                arrays = tree.arrays(library="np")
                matched_energy = arrays["matched_energy"]
                mc_energy = arrays["mc_energy"]
                mc_energy = np.asarray(mc_energy).reshape(-1)
                matched_energy = np.asarray(matched_energy).reshape(-1)
                response = matched_energy / mc_energy
                q16, q84 = np.percentile(response, [16, 84])
                median = np.median(response)
                sigma = (q84 - q16)/2
                resolution = sigma / median
                results["baseline_median"][a].append(median)
                results["baseline_low"][a].append(median-q16)
                results["baseline_high"][a].append(q84-median)
                results["baseline_resolution"][a].append(resolution)

                file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_calibrated_base.root")
                tree = file["MatchedClusters"]
                arrays = tree.arrays(library="np")
                matched_energy = arrays["matched_energy"]
                mc_energy = arrays["mc_energy"]
                mc_energy = np.asarray(mc_energy).reshape(-1)
                matched_energy = np.asarray(matched_energy).reshape(-1)
                response = matched_energy / mc_energy
                q16, q84 = np.percentile(response, [16, 84])
                median = np.median(response)
                sigma = (q84 - q16)/2
                resolution = sigma / median
                results["4cal_median"][a].append(median)
                results["4cal_low"][a].append(median-q16)
                results["4cal_high"][a].append(q84-median)
                results["4cal_resolution"][a].append(resolution)
                if (pid == 211):
                    ecal_barrel = arrays["ecal_barrel"]
                    hcal_barrel = arrays["hcal_barrel"]
                    ecal_endcap = arrays["ecal_endcap"]
                    hcal_endcap = arrays["hcal_endcap"]
                    ecal_barrel = np.asarray(ecal_barrel).reshape(-1)
                    hcal_barrel = np.asarray(hcal_barrel).reshape(-1)
                    ecal_endcap = np.asarray(ecal_endcap).reshape(-1)
                    hcal_endcap = np.asarray(hcal_endcap).reshape(-1)
                    ecal_ratio = ecal_barrel + ecal_endcap
                    hcal_ratio = hcal_barrel + hcal_endcap
                    total = ecal_ratio + hcal_ratio
                    ecal_ratio = ecal_ratio / total
                    hcal_ratio = hcal_ratio / total

                    for i in range(len(response)):
                        baseline = response[i]
                        correction_1 = (ecal_endcap[i] / total[i]) * .09
                        correction_2 =  (hcal_endcap[i] / total[i]) * .03 
                        correction_3 = (ecal_barrel[i] / total[i]) * .16
                        correction_4 =  (hcal_barrel[i]  / total[i]) * .19
                        new_fraction = baseline - correction_1 + correction_2 - correction_3 + correction_4
                        response[i] = new_fraction

                    q16, q84 = np.percentile(response, [16, 84])
                    median = np.median(response)
                    sigma = (q84 - q16)/2
                    resolution = sigma / median
                    results["4calhitcal_median"][a].append(median)
                    results["4calhitcal_low"][a].append(median-q16)
                    results["4calhitcal_high"][a].append(q84-median)
                    results["4calhitcal_resolution"][a].append(resolution)
            energies = [10, 30, 50]
    # ------------------ PLOTTING SECTION ------------------
# (Place this inside your 'for pid in particle:' loop right after data harvesting)

    plt.figure(figsize=(6.4, 4.8))

    # Define the configurations we want to plot for the current particle
    # Each tuple contains: (dictionary_prefix, config_key, label_string)
    configs = [
        ("baseline", "baseline", "Baseline"),
        ("4cal", "4cal", "Regional Scale")
    ]

    # If it's a pion, dynamically append the 3rd dataset configuration
    if pid == 211:
        configs.append(("4calhitcal", "4calhitcal", "Longitudinal Profile"))
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    # Loop through our detector regions (Endcap and Barrel)
    for a in angle:
        region_label = "Endcap" if a == 15 else "Barrel"
        
        # Loop through each algorithm configuration dynamically using your dicts
        for prefix, key, label_base in configs:
            
            # Build the dynamic dictionary keys for your results
            median_key = f"{prefix}_median"
            low_key = f"{prefix}_low"
            high_key = f"{prefix}_high"
            
            # Pull styling parameters from your configuration dictionaries
            color = geom_colors[a]
            marker = markers[key]
            linestyle = line_styles[key]
            fmt = f"{marker}{linestyle}"
            
            # Plot the errorbar using your automated styles
            plt.errorbar(
                energies, 
                results[median_key][a],
                yerr=[results[low_key][a], results[high_key][a]],
                fmt=fmt, 
                color=color, 
                capsize=4, 
                label=f"{region_label} {label_base}"
            )

    # Canvas Aesthetics (matching your reference image exactly)
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Median Response")
    #plt.title(f"Response vs Energy {particle_labels[pid]} Multi-Cluster BIB")

    # Bottom right legend, no grid lines, clean white background frame
    plt.ylim(0,1.6)
    plt.legend(loc="lower right", frameon=False) 
    plt.tight_layout()
    plt.savefig(f"newresponse_vs_energy_pid{pid}_multiclusterbib.pdf", dpi=300)
    plt.close()

        # ----------------- RESOLUTION PLOTTING SECTION -----------------
    # (Place this inside your 'for pid in particle:' loop right after the response plot)

    plt.figure(figsize=(6.4, 4.8))

    # Define the configurations we want to plot for the current particle
    configs = [
        ("baseline", "baseline", "Baseline"),
        ("4cal", "4cal", "Regional Scale")
    ]

    # Dynamically append the 3rd dataset configuration if it's a pion
    if pid == 211:
        configs.append(("4calhitcal", "4calhitcal", "Longitudinal Profile"))
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    # Loop through our detector regions (Endcap and Barrel)
    for a in angle:
        region_label = "Endcap" if a == 15 else "Barrel"
        
        # Loop through each algorithm configuration dynamically using your dicts
        for prefix, key, label_base in configs:
            
            # Build the dynamic dictionary key for your resolution data
            res_key = f"{prefix}_resolution"
            
            # Pull styling parameters from your dictionaries
            color = geom_colors[a]
            marker = markers[key]
            linestyle = line_styles[key]
            fmt = f"{marker}{linestyle}"
            
            # Plot the resolution line
            plt.plot(
                energies, 
                results[res_key][a],
                fmt, 
                color=color, 
                label=f"{region_label} {label_base} "
            )

    # Canvas Aesthetics
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Resolution")
    #plt.title(f"Resolution vs Energy {particle_labels[pid]} Multi-Cluster BIB")

    # Clean layout matching your style requirements
    plt.legend(loc="upper right", frameon=False) 
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"xnewresolution_vs_energy_pid{pid}_multiclusterbib.pdf", dpi=300)
    plt.close()

print(f"\nParticle PID: {pid}")

for a in angle:
    region = "Endcap" if a == 15 else "Barrel"

    print(f"\n{region} ({a} deg)")
    print("Energies:", energies)
    print("Baseline Resolution:", results["baseline_resolution"][a])
    print("Regional Scale Resolution:", results["4cal_resolution"][a])

    if pid == 211:
        print("Longitudinal Profile Resolution:", results["4calhitcal_resolution"][a])