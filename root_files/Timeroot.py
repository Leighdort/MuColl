#Timeroot.py

#I will find a way per energy to look at hits log time 
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import json
import uproot
#import ROOT

energies = [10, 50, 100, 150, 200]

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec"
}

for num in energies:
    file = uproot.open(f"reco_outputp{num}.edm4hep.root")
    events = file["events"]

    times_per_system = {name: [] for name in system2name.values()}

    for name in system2name.values():
        prefix = f"{name}/{name}"
        times_array = events[f"{prefix}.time"].array()
        for t in times_array:
            t_np = np.array(t)
            times_per_system[name].extend(t_np)

    # Combine barrel + endcap
    ecal_times = np.array(times_per_system["EcalBarrelCollectionRec"] +
                          times_per_system["EcalEndcapCollectionRec"])
    hcal_times = np.array(times_per_system["HcalBarrelCollectionRec"] +
                          times_per_system["HcalEndcapCollectionRec"])

    # Keep only positive times
    ecal_times = ecal_times[ecal_times > 0]
    hcal_times = hcal_times[hcal_times > 0]

    # ECAL histogram
# Make log-spaced bins
    bins = np.logspace(np.log10(ecal_times.min()), np.log10(ecal_times.max()), 50)
    plt.hist(ecal_times, bins=bins, alpha=0.7, color='blue')
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.title(f"ECAL time distribution, {num} GeV")
    plt.xscale('log')  # keep axis in actual time units
    plt.tight_layout()
    plt.savefig(f"time_hits_ecal{num}p.pdf")
    plt.close()

    bins = np.logspace(np.log10(hcal_times.min()), np.log10(hcal_times.max()), 50)
    plt.hist(hcal_times, bins=bins, alpha=0.7, color='blue')
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.title(f"HCAL time distribution, {num} GeV")
    plt.xscale('log')  # keep axis in actual time units #you're logging the values
    plt.tight_layout()
    plt.savefig(f"time_hits_hcal{num}p.pdf")
    plt.close()
