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

#energies = [1, 2, 5, 10, 50, 100, 150, 200]
energies = [1, 2, 5, 10]
#energies = [50,100]
#energies = [150]
#energies = [200]

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
'''
# ======= ELECTRONS =======
elec_ecal_end_mean = []
elec_hcal_start_mean = []
elec_ecal_end_std = []
elec_hcal_start_std = []

'''
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec",
                "EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
'''
for num in energies:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]

    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all = pandora_clusters["PandoraClusters.hits_end"].array()

    ecal_times = []
    hcal_times = []

    # Preload times
    times = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        times[name] = events[f"{prefix}.time"].array()

    # Event-level max/min arrays
    event_ecal_max_times = []
    event_hcal_min_times = []

    for i in range(events.num_entries):
        if i % 1000 == 0:
            print(f"Event {i}")

        # Minimal fix: initialize per-event arrays here
        ecal_times_event = []
        hcal_times_event = []

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr = hits_end_all[i]
        hit_index = hit_index_all[i]
        collection_ID = collectionID_all[i]

        for j in range(len(hits_begin_arr)):
            lo = hits_begin_arr[j]
            hi = hits_end_arr[j]
            idxs = hit_index[lo:hi]
            sysIDs = collection_ID[lo:hi]

            if len(idxs) == 0:
                continue

            sysnames = np.vectorize(system2name.get)(sysIDs)

            if 'None' in sysnames.astype(str):
                print(f"Skipping cluster {j} in event {i} at energy {num} GeV due to unknown system ID")
                continue

            mask = (sysnames != "Skip")
            sysnames = sysnames[mask]
            idxs = idxs[mask]

            if len(sysnames) == 0:
                continue

            for name, idx in zip(sysnames, idxs):
                if "Ecal" in name:
                    ecal_times.append(times[name][i][idx])
                    ecal_times_event.append(times[name][i][idx])
                elif "Hcal" in name:
                    hcal_times.append(times[name][i][idx])
                    hcal_times_event.append(times[name][i][idx])  # minimal fix: typo corrected

        # Minimal fix: compute event-level max/min **after clusters**
        if len(ecal_times_event) > 0:
            event_ecal_max_times.append(np.max(ecal_times_event))
        if len(hcal_times_event) > 0:
            event_hcal_min_times.append(np.min(hcal_times_event))

    # Histogram and summary
    ecal_times = np.array(ecal_times)
    hcal_times = np.array(hcal_times)
    ecal_times = ecal_times[ecal_times > 0]
    hcal_times = hcal_times[hcal_times > 0]

    mean_ecal = np.mean(event_ecal_max_times)
    std_ecal = np.std(event_ecal_max_times)
    mean_hcal = np.mean(event_hcal_min_times)
    std_hcal = np.std(event_hcal_min_times)
    print(f"electrons {num}, meanecal= {mean_ecal}, stdecal = {std_ecal}")
    print(f"electrons {num}, meanhcal = {mean_hcal}, stdhcal = {std_hcal}")

    if len(ecal_times) == 0 or len(hcal_times) == 0:
        print(f"Skipping hist for {num} GeV: No ECAL or HCAL times.")
        continue

    bins = np.logspace(np.log10(ecal_times.min()), np.log10(hcal_times.max()), 100)
    plt.hist(ecal_times, bins=bins, alpha=0.6, color='red', label="ecal_times")
    plt.hist(hcal_times, bins=bins, alpha=0.6, color='blue', label="hcal_times")
    plt.xlabel("Times")
    plt.ylabel("Counts")
    plt.title(f"ECAL and HCAL, {num} GeV Electrons for Clusters")
    plt.xscale('log')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"cluster_time_hits{num}e10x.pdf")
    plt.close()
'''


# ======= PIONS =======
pion_ecal_end_mean = []
pion_hcal_start_mean = []
pion_ecal_end_std = []
pion_hcal_start_std = []

for num in energies:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]

    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all = pandora_clusters["PandoraClusters.hits_end"].array()

    ecal_times = []
    hcal_times = []

    # Preload times
    times = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        times[name] = events[f"{prefix}.time"].array()

    # Event-level max/min arrays
    event_ecal_max_times = []
    event_hcal_min_times = []

    for i in range(events.num_entries):
        if i % 1000 == 0:
            print(f"Event {i}")

        # Minimal fix: initialize per-event arrays here
        ecal_times_event = []
        hcal_times_event = []

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr = hits_end_all[i]
        hit_index = hit_index_all[i]
        collection_ID = collectionID_all[i]

        for j in range(len(hits_begin_arr)):
            lo = hits_begin_arr[j]
            hi = hits_end_arr[j]
            idxs = hit_index[lo:hi]
            sysIDs = collection_ID[lo:hi]

            if len(idxs) == 0:
                continue

            sysnames = np.vectorize(system2name.get)(sysIDs)

            if 'None' in sysnames.astype(str):
                print(f"Skipping cluster {j} in event {i} at energy {num} GeV due to unknown system ID")
                continue

            mask = (sysnames != "Skip")
            sysnames = sysnames[mask]
            idxs = idxs[mask]

            if len(sysnames) == 0:
                continue

            for name, idx in zip(sysnames, idxs):
                if "Ecal" in name:
                    ecal_times.append(times[name][i][idx])
                    ecal_times_event.append(times[name][i][idx])
                elif "Hcal" in name:
                    hcal_times.append(times[name][i][idx])
                    hcal_times_event.append(times[name][i][idx])

        # Minimal fix: compute event-level max/min **after clusters**
        if len(ecal_times_event) > 0:
            event_ecal_max_times.append(np.max(ecal_times_event))
        if len(hcal_times_event) > 0:
            event_hcal_min_times.append(np.min(hcal_times_event))

    # Histogram and summary
    ecal_times = np.array(ecal_times)
    hcal_times = np.array(hcal_times)
    ecal_times = ecal_times[ecal_times > 0]
    hcal_times = hcal_times[hcal_times > 0]

    mean_ecal = np.mean(event_ecal_max_times)
    std_ecal = np.std(event_ecal_max_times)
    mean_hcal = np.mean(event_hcal_min_times)
    std_hcal = np.std(event_hcal_min_times)
    print(f"pion {num}, mean_ecal = {mean_ecal}, std_ecal = {std_ecal}, mean_hcal = {mean_hcal}, std_hcal = {std_hcal}")
    if len(ecal_times) == 0 or len(hcal_times) == 0:
        print(f"Skipping hist for {num} GeV: No ECAL or HCAL times.")
        continue

    bins = np.logspace(np.log10(ecal_times.min()), np.log10(hcal_times.max()), 100)
    plt.hist(ecal_times, bins=bins, alpha=0.6, color='red', label="ecal_times")
    plt.hist(hcal_times, bins=bins, alpha=0.6, color='blue', label="hcal_times")
    plt.xlabel("Times")
    plt.ylabel("Counts")
    plt.title(f"ECAL and HCAL, {num} GeV Pions for Clusters")
    plt.xscale('log')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"cluster_time_hits{num}p10x.pdf")
    plt.close()



'''
#Now we are going to graph everything together
#This might not show everything
plt.errorbar(energies, elec_ecal_end_mean, yerr=elec_ecal_end_std, fmt='o', capsize=4, label="Electron Average End of Ecal")
plt.errorbar(energies, elec_hcal_start_mean, yerr=elec_hcal_start_std, fmt='o', capsize = 4, label="Electron Average Start of Ecal")
plt.errorbar(energies, pion_ecal_end_mean, yerr=pion_ecal_end_std, fmt='o', capsize=4, label="Pion Average End of Ecal")
plt.errorbar(energies, pion_hcal_start_mean, yerr=pion_hcal_start_std, fmt='o', capsize = 4, label="Pion Average Start of Ecal")
plt.xlabel("Beam Energy")
plt.ylabel("Time ")
plt.title("Average Start/End Hcal/Ecal for Pions & Electrons")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_cluster_time.pdf")
plt.close()
'''