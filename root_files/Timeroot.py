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
    file = uproot.open(f"reco_outpute{num}.edm4hep.root")
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
    bins = np.logspace(np.log10(ecal_times.min()), np.log10(hcal_times.max()), 100)
    plt.hist(ecal_times, bins=bins, alpha = 0.6, color = 'red', label = "ecal_times")
    plt.hist(hcal_times, bins=bins, alpha = 0.6, color = 'blue', label= "hcal_times")
    plt.xlabel("Times")
    plt.ylabel("Counts")
    plt.title(f"ECAL and HCAL, {num} GeV Electrons ")
    plt.xscale('log')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"time_hits{num}e.pdf")
    plt.close()
#Now we are going to have 2 different y-axis
    fig, ax1 = plt.subplots()
    h1 = ax1.hist(ecal_times, bins=bins, alpha=0.6, color='red', label = "ecal_times")
    ax1.set_xlabel("Times")
    ax1.set_ylabel("ECAL Counts", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xscale('log')
    ax2 = ax1.twinx()
    h2 = ax2.hist(hcal_times, bins=bins, alpha = 0.6, color = 'blue', label= "hcal_times")
    ax2.set_ylabel("HCAL Counts", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title(f"ECAL and HCAL, {num} GeV Electrons")
    fig.tight_layout()
    plt.savefig(f"time_hits_rescale{num}e.pdf")
    plt.close()
#We are going to have 2 different y-axis but have the focus region be 0 to 20
    fig, ax1 = plt.subplots()
    h1 = ax1.hist(ecal_times, bins=bins, alpha=0.6, color='red', label = "ecal_times")
    ax1.set_xlabel("Times")
    ax1.set_ylabel("ECAL Counts", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xscale('log')
    ax2 = ax1.twinx()
    h2 = ax2.hist(hcal_times, bins=bins, alpha = 0.6, color = 'blue', label= "hcal_times")
    ax2.set_ylabel("HCAL Counts", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title(f"ECAL and HCAL, {num} GeV Electrons")
    fig.tight_layout()
    ax1.set_xlim(0.9, 20)
    plt.savefig(f"time_hits_rescale_lim{num}e.pdf")
    plt.close() 

    '''
    plt.hist(ecal_times, bins=bins, alpha=0.7, color='blue')
    plt.hist
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.title(f"ECAL time distribution, {num} GeV Electrons")
    plt.xscale('log')  # keep axis in actual time units
    plt.tight_layout()
    plt.savefig(f"time_hits{num}3.pdf")
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
'''