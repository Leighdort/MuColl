#This is all things time
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

energies = [10, 50, 100, 150, 200]
#outputelectron_reco10_hits.csv
average_time = []
stdev_time = []
average_start = []
stdev_start = []
average_end = []
stdev_end = []
average_difference = []
stdev_difference = []
for energy in energies:
    average_per_event = []
    maxes = []
    mins = []
    differences = []
    df = pd.read_csv(f"outputelectron_reco{energy}_hits.csv")
    for event_id, event_group in df.groupby("event"):
        avg_time = event_group["time"].mean()
        average_per_event.append(avg_time)
        start = event_group["time"].min()
        end = event_group["time"].max()
        mid = end - start
        mins.append(start)
        maxes.append(end)
        differences.append(mid)
    average_time.append(np.mean(average_per_event))
    stdev_time.append(np.std(average_per_event))
    average_start.append(np.mean(mins))
    average_end.append(np.mean(maxes))
    average_difference.append(np.mean(differences))
    stdev_start.append(np.std(mins))
    stdev_end.append(np.std(maxes))
    stdev_difference.append(np.std(differences))

#plotting time stuff        
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.errorbar(energies, average_end, yerr=stdev_end, fmt='o-', capsize=5, label='Max')
plt.title("Max (End) Time vs Energy")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Max Time")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.errorbar(energies, average_start, yerr=stdev_start, fmt='o-', capsize=5, label='Min')
plt.title("Min (Start) Time vs Energy")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Min Time")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.errorbar(energies, average_time, yerr=stdev_time, fmt='o-', capsize=5, label='Mean')
plt.title("Mean Time vs Energy")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Mean Time")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.errorbar(energies, average_difference, yerr=stdev_difference, fmt='o-', capsize=5, label='Range')
plt.title("Time Range (End − Start) vs Energy")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Difference (End − Start)")
plt.grid(True)

plt.tight_layout()

plt.savefig("electron_time_plots.pdf", bbox_inches="tight")

plt.show()

#Ok Now I am going to look at time distribution in an energy bin
#_____________________________________________________________________________________________
energies = [10, 50, 100, 150, 200]

#Bins are layers, hits per layer, can histogram 
#Do it for a single event, time distribution per energy bin
#Then do it in 1 energy bin time hits? 

for energy in energies:
    df = pd.read_csv(f"outputpion_reco{energy}_hits.csv")
    all_times = df["time"]
    negative_times = all_times[all_times < 0]
    positive_times = all_times[all_times >= 0]
    print(f"Energy {energy} MeV — Negative times: {len(negative_times)} hits")
    top = len(negative_times)
    bottom = len(df)
    print(top/bottom)
    plt.figure()
    bins = np.logspace(-3, np.log10(40), 100)
    #bins = np.linspace(0, 40, 100)  # 100 bins from 0 to 40
    plt.hist(positive_times, bins=bins, alpha=0.5, label=f"{energy} MeV")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(f"pionhistogramtime{energy}.pdf")
    plt.close()
    print("Histogram made")

#Locations to look at:
system2name = {
    20: "ecal barrel",
    29: "ecal endcap",
    10: "hcal barrel",
    11: "hcal barrel",
    13: "yoke barrel",
    14: "yoke endcap"
    #Problem some are 13 and 14
}

for energy in energies:
    df = pd.read_csv(f"outputpion_reco{energy}_hits.csv")
    system = df["system"]
    ecal_mask = (system == "ecal barrel") | (system == "ecal endcap")
    ecal_system = df[ecal_mask]
    ecal_times = ecal_system["time"]
    hcal_mask = (system == "hcal barrel") | (system == "hcal endcap")
    hcal_system = df[hcal_mask]
    hcal_times = hcal_system["time"]
    hcal_times = hcal_times[hcal_times >= 0]
    ecal_times = ecal_times[ecal_times >= 0]
    plt.figure()
    bins = np.logspace(-3, np.log10(40), 100)
    plt.hist(hcal_times, bins=bins, alpha=0.5, label=f"{energy} MeV")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(f"pion_hcal_histogramtime{energy}.pdf")
    plt.close()
    plt.figure()
    plt.hist(ecal_times, bins=bins, alpha=0.5, label=f"{energy} MeV")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(f"pion_ecal_histogramtime{energy}.pdf")
    plt.close()
    

df = pd.read_csv(f"outputpion_reco100_hits.csv")
mask = df["event"] == 10
df = df[mask]
system = df["system"]
ecal_mask = (system == "ecal barrel") | (system == "ecal endcap")
ecal_system = df[ecal_mask]
ecal_times = ecal_system["time"]
hcal_mask = (system == "hcal barrel") | (system == "hcal endcap")
hcal_system = df[hcal_mask]
hcal_times = hcal_system["time"]
hcal_times = hcal_times[hcal_times >= 0]
ecal_times = ecal_times[ecal_times >= 0]
plt.figure()
bins = np.logspace(-3, np.log10(40), 100)
plt.hist(hcal_times, bins=bins, alpha=0.5, label=f"100 MeV")
plt.legend()
plt.xscale("log")
plt.xlabel("Time")
plt.ylabel("Counts")
plt.tight_layout()
plt.savefig(f"pion_hcal_histogramtime100.10.pdf")
plt.close()
plt.figure()
plt.hist(ecal_times, bins=bins, alpha=0.5, label=f"100 MeV")
plt.legend()
plt.xscale("log")
plt.xlabel("Time")
plt.ylabel("Counts")
plt.tight_layout()
plt.savefig(f"pion_ecal_histogramtime100.10.pdf")
plt.close()
