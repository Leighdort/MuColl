#Bib response and resolution
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

angle = [15, 85]
particle = [211]
energy = [2, 5, 10, 30, 50]
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
    summary_response = {
    15: {},
    85: {}
    }

    summary_resolution = {
    15: {},
    85: {}
    }
    for num in energy: 
        for a in angle: 
            #Open up the file
            #Find resolution
            #Graph gaussian
            file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_uncalibrated.root")
            tree = file["MatchedClusters"]
            arrays = tree.arrays(library="np")
            matched_energy = arrays["matched_energy"]
            mc_energy = arrays["mc_energy"]
            mc_energy = np.asarray(mc_energy).reshape(-1)
            matched_energy = np.asarray(matched_energy).reshape(-1)
            print(type(matched_energy))
            print(matched_energy.shape)
            print(matched_energy.dtype)
            print(matched_energy[0])  # what does one entry look like?
            #matched_energy = np.array(matched_energy, dtype=float)
            #mc_energy = np.array(mc_energy, dtype=float)
            response = matched_energy / mc_energy 
            print(type(matched_energy))
            print(matched_energy.shape)
            print(matched_energy.dtype)
            print(matched_energy[0]) 
            q16, q84 = np.percentile(response, [16, 84])
            median = np.median(response)
            sigma = (q84 - q16)/2
            resolution = sigma / median
            results["baseline_median"][a].append(median)
            results["baseline_low"][a].append(median-q16)
            results["baseline_high"][a].append(q84-median)
            results["baseline_resolution"][a].append(resolution)
            if num == 50:
                summary_response[a]["nobib_uncal"] = {
                    "median": median,
                    "low": median - q16,
                    "high": q84 - median
                }
                summary_resolution[a]["nobib_uncal"] = resolution
                
                file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_software_on.root")
                tree = file["MatchedClusters"]
                arrays = tree.arrays(library="np")
                matched_energy = arrays["matched_energy"]
                mc_energy = arrays["mc_energy"]
                mc_energy = np.asarray(mc_energy).reshape(-1)
                matched_energy = np.asarray(matched_energy).reshape(-1)
                #print(type(matched_energy))
                #print(matched_energy.shape)
                #print(matched_energy.dtype)
                #print(matched_energy[0])  # what does one entry look like?
                #matched_energy = np.array(matched_energy, dtype=float)
                #mc_energy = np.array(mc_energy, dtype=float)
                response = matched_energy / mc_energy 
                #print(type(matched_energy))
                #print(matched_energy.shape)
                #print(matched_energy.dtype)
                #print(matched_energy[0]) 
                q16, q84 = np.percentile(response, [16, 84])
                median = np.median(response)
                sigma = (q84 - q16)/2
                resolution = sigma / median
                summary_response[a]["bib_uncal"] = {
                    "median": median,
                    "low": median - q16,
                    "high": q84 - median
                }
                summary_resolution[a]["bib_uncal"] = resolution
            

            file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_calibrated.root")
            tree = file["MatchedClusters"]
            arrays = tree.arrays(library="np")
            matched_energy = arrays["matched_energy"]
            mc_energy = arrays["mc_energy"]
            mc_energy = np.asarray(mc_energy).reshape(-1)
            matched_energy = np.asarray(matched_energy).reshape(-1)
            print(type(matched_energy))
            print(matched_energy.shape)
            print(matched_energy.dtype)
            print(matched_energy[0])  # what does one entry look like?
            #matched_energy = np.array(matched_energy, dtype=float)
            #mc_energy = np.array(mc_energy, dtype=float)
            response = matched_energy / mc_energy 
            print(type(matched_energy))
            print(matched_energy.shape)
            print(matched_energy.dtype)
            print(matched_energy[0]) 
            q16, q84 = np.percentile(response, [16, 84])
            median = np.median(response)
            sigma = (q84 - q16)/2
            resolution = sigma / median
            results["4cal_median"][a].append(median)
            results["4cal_low"][a].append(median-q16)
            results["4cal_high"][a].append(q84-median)
            results["4cal_resolution"][a].append(resolution)
            if num == 50:
                summary_response[a]["nobib_4cal"] = {
                    "median": median,
                    "low": median - q16,
                    "high": q84 - median
                }
                summary_resolution[a]["nobib_4cal"] = resolution
                file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_calib10.root")
                tree = file["MatchedClusters"]
                arrays = tree.arrays(library="np")
                matched_energy = arrays["matched_energy"]
                mc_energy = arrays["mc_energy"]
                mc_energy = np.asarray(mc_energy).reshape(-1)
                matched_energy = np.asarray(matched_energy).reshape(-1)
                print(type(matched_energy))
                print(matched_energy.shape)
                print(matched_energy.dtype)
                print(matched_energy[0])  # what does one entry look like?
                #matched_energy = np.array(matched_energy, dtype=float)
                #mc_energy = np.array(mc_energy, dtype=float)
                response = matched_energy / mc_energy 
                print(type(matched_energy))
                print(matched_energy.shape)
                print(matched_energy.dtype)
                print(matched_energy[0]) 
                q16, q84 = np.percentile(response, [16, 84])
                median = np.median(response)
                sigma = (q84 - q16)/2
                resolution = sigma / median
                summary_response[a]["bib_4cal"] = {
                    "median": median,
                    "low": median - q16,
                    "high": q84 - median
                }
                summary_resolution[a]["bib_4cal"] = resolution
            if pid == 211:
                file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_calibrated.root")
                tree = file["MatchedClusters"]
                arrays = tree.arrays(library="np")
                matched_energy = arrays["matched_energy"]
                mc_energy = arrays["mc_energy"]
                mc_energy = np.asarray(mc_energy).reshape(-1)
                matched_energy = np.asarray(matched_energy).reshape(-1)
                print(type(matched_energy))
                print(matched_energy.shape)
                print(matched_energy.dtype)
                print(matched_energy[0])  # what does one entry look like?
                #matched_energy = np.array(matched_energy, dtype=float)
                #mc_energy = np.array(mc_energy, dtype=float)
                response = matched_energy / mc_energy 
                print(type(matched_energy))
                print(matched_energy.shape)
                print(matched_energy.dtype)
                print(matched_energy[0]) 
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
                if num == 50:
                    summary_response[a]["nobib_hitcal"] = {
                        "median": median,
                        "low": median - q16,
                        "high": q84 - median
                    }
                    summary_resolution[a]["nobib_hitcal"] = resolution
                if num == 50:
                    file = uproot.open(f"matched_clusters_pid{pid}_E{num}_theta{a}_calib10.root")
                    tree = file["MatchedClusters"]
                    arrays = tree.arrays(library="np")
                    matched_energy = arrays["matched_energy"]
                    mc_energy = arrays["mc_energy"]
                    mc_energy = np.asarray(mc_energy).reshape(-1)
                    matched_energy = np.asarray(matched_energy).reshape(-1)
                    print(type(matched_energy))
                    print(matched_energy.shape)
                    print(matched_energy.dtype)
                    print(matched_energy[0])  # what does one entry look like?
                    #matched_energy = np.array(matched_energy, dtype=float)
                    #mc_energy = np.array(mc_energy, dtype=float)
                    response = matched_energy / mc_energy 
                    print(type(matched_energy))
                    print(matched_energy.shape)
                    print(matched_energy.dtype)
                    print(matched_energy[0]) 
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


                    #Now we want the Multi-Cluster gaussians

                    
                    q16, q84 = np.percentile(response, [16, 84])
                    median = np.median(response)
                    sigma = (q84 - q16)/2
                    resolution = sigma / median
                    if num == 50:
                        summary_response[a]["bib_hitcal"] = {
                            "median": median,
                            "low": median - q16,
                            "high": q84 - median
                        }
                        summary_resolution[a]["bib_hitcal"] = resolution

    #Now i want to plot these 3 things on a graph with energy as x axis and resolution w/ error or response as y axis
    #two separate graphs with labels 
    #Electrons have no 4calhitcal
    energies = [2, 5, 10, 30, 50]
    
    # ==========================================
    # RESPONSE PLOT
    # ==========================================

    for angle_val, region in zip([15, 85], ["Endcap", "Barrel"]):
        plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
        # Baseline
        plt.errorbar(
            energies,
            results["baseline_median"][angle_val],
            yerr=[
                results["baseline_low"][angle_val],
                results["baseline_high"][angle_val]
            ],
            color=geom_colors[angle_val],
            linestyle=line_styles["baseline"],
            marker=markers["baseline"],
            capsize=4, 
            label=f"{region} Baseline"
        )

        # 4Cal
        plt.errorbar(
            energies,
            results["4cal_median"][angle_val],
            yerr=[
                results["4cal_low"][angle_val],
                results["4cal_high"][angle_val]
            ],
            color=geom_colors[angle_val],
            linestyle=line_styles["4cal"],
            marker=markers["4cal"],
            capsize=4, 
            label=f"{region} Regional Scale"
        )

        # 4CalHitCal only for pions
        if pid == 211:
            plt.errorbar(
                energies,
                results["4calhitcal_median"][angle_val],
                yerr=[
                    results["4calhitcal_low"][angle_val],
                    results["4calhitcal_high"][angle_val]
                ],
                color=geom_colors[angle_val],
                linestyle=line_styles["4calhitcal"],
                marker=markers["4calhitcal"],
                capsize=4, 
                label=f"{region} Longitudinal Profile"
            )

    plt.xlabel("Energy [GeV]")
    plt.ylabel("Median Response")
    #plt.title(f"Response vs Energy {particle_labels[pid]} Multi-Cluster")

    #plt.grid(True, alpha=0.3)
    #plt.ylim(0,1.2)
    plt.ylim(0, 1.6)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(f"newresponse_vs_energy_pid{pid}_multicluster.pdf")
    plt.close()

        # ==========================================
    # RESOLUTION PLOT
    # ==========================================

    for angle_val, region in zip([15, 85], ["Endcap", "Barrel"]):

        # Baseline
        plt.plot(
            energies,
            results["baseline_resolution"][angle_val],
            color=geom_colors[angle_val],
            linestyle=line_styles["baseline"],
            marker=markers["baseline"],
            label=f"{region} Baseline"
        )

        # 4Cal
        plt.plot(
            energies,
            results["4cal_resolution"][angle_val],
            color=geom_colors[angle_val],
            linestyle=line_styles["4cal"],
            marker=markers["4cal"], 
            label=f"{region} Regional Scale"
        )

        # 4CalHitCal
        if pid == 211:
            plt.plot(
                energies,
                results["4calhitcal_resolution"][angle_val],
                color=geom_colors[angle_val],
                linestyle=line_styles["4calhitcal"],
                marker=markers["4calhitcal"],
                label=f"{region} Longitudinal Profile"
            )

    plt.xlabel("Energy [GeV]")
    plt.ylabel("Resolution")
    #plt.title(f"Resolution vs Energy {particle_labels[pid]} Multi-Cluster")

    #plt.grid(True, alpha=0.3)
    #plt.ylim(0, 0.65)
    plt.ylim(0, 1)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(f"xnewresolution_vs_energy_pid{pid}_multicluster.pdf")
    plt.close()

            
