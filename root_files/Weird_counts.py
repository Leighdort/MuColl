#Checking weird counts

# Checking weird counts

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
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}

real_systems = [
    "EcalBarrelCollectionRec",
    "HcalBarrelCollectionRec",
    "EcalEndcapCollectionRec",
    "HcalEndcapCollectionRec"
]

energies = [1, 2, 5, 10, 50, 100, 150, 200]

############################################################
# ELECTRONS
############################################################

elecweird = []
elecnormal = []
elecclusters_total = []
elecpolluted_total = []
elecpolluted_events = []
for num in energies:
    count_weird = 0
    count_total = 0
    clusters = 0
    polluted = 0
    polluted_events = 0

    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]

    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()

    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

    for i in range(events.num_entries):
        if (i % 1000 == 0):
            print(i)
        polluted_event = False
        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]
        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]

        for j in range(len(hits_begin_arr)):
            polluted_cluster = False
            clusters += 1

            start = hits_begin_arr[j]
            end   = hits_end_arr[j]

            indices = hit_index[start:end]
            ids     = collection_ID[start:end]

            count_total += len(ids)

            if len(indices) == 0:
                continue

            for sysid in ids:
                sysname = system2name[sysid]
                if sysname == "Skip":
                    count_weird += 1
                    polluted_cluster = True

            if polluted_cluster:
                polluted_event = True

                polluted += 1
        if polluted_event == True:
            polluted_events += 1
    elecpolluted_events.append(polluted_events)
    elecclusters_total.append(clusters)
    elecpolluted_total.append(polluted)
    elecweird.append(count_weird)
    elecnormal.append(count_total)

elecweird = np.array(elecweird)
elecnormal = np.array(elecnormal)
elecpolluted_events = np.array (elecpolluted_events)
elec_ratio_hits = elecweird / elecnormal
elecclusters_total = np.array(elecclusters_total)
elecpolluted_total = np.array(elecpolluted_total)
elec_ratio_clusters = elecpolluted_total / elecclusters_total


############################################################
# PIONS
############################################################

pionweird = []
pionnormal = []
pionclusters_total = []
pionpolluted_total = []
pionpolluted_events = []
for num in energies:
    count_weird = 0
    count_total = 0
    clusters = 0
    polluted = 0
    pollutedevents = 0
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]

    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()

    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

    for i in range(events.num_entries):
        if (i % 1000 == 0):
            print(i)
        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]
        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]
        event_polluted = False

        for j in range(len(hits_begin_arr)):
            polluted_cluster = False
            clusters += 1

            start = hits_begin_arr[j]
            end   = hits_end_arr[j]

            indices = hit_index[start:end]
            ids     = collection_ID[start:end]

            count_total += len(ids)

            if len(indices) == 0:
                continue

            for sysid in ids:
                sysname = system2name[sysid]
                if sysname == "Skip":
                    count_weird += 1
                    polluted_cluster = True

            if polluted_cluster:
                polluted += 1
                event_polluted = True
        if event_polluted == True:
            pollutedevents +=1
    pionclusters_total.append(clusters)
    pionpolluted_total.append(polluted)
    pionweird.append(count_weird)
    pionnormal.append(count_total)
    pionpolluted_events.append(pollutedevents)

pionweird = np.array(pionweird)
pionnormal = np.array(pionnormal)
pionpolluted_events = np.array(pionpolluted_events)
pion_ratio_hits = pionweird / pionnormal
pionclusters_total = np.array(pionclusters_total)
pionpolluted_total = np.array(pionpolluted_total)
pion_ratio_clusters = pionpolluted_total / pionclusters_total

############################################################
# PLOTTING
############################################################

# Weird hit counts
plt.figure()
plt.plot(energies, pionweird, marker='o', label="Pion")
plt.plot(energies, elecweird, marker='o', label="Electron")
plt.xlabel("Energy")
plt.ylabel("Punch Through Hit Count")
plt.title("Punch Through Hits vs Energy")
plt.grid(True)
plt.legend()
plt.savefig("punchhits_both.pdf")
plt.close()

# Hit ratios
plt.figure()
plt.plot(energies, pion_ratio_hits, marker='o', label="Pion")
plt.plot(energies, elec_ratio_hits, marker='o', label="Electron")
plt.xlabel("Energy")
plt.ylabel("Punch Through Hits / Total Hits")
plt.title("Punch Through Hit Ratio")
plt.grid(True)
plt.legend()
plt.savefig("ratiohits_both.pdf")
plt.close()

# Cluster counts polluted
plt.figure()
plt.plot(energies, pionpolluted_total, marker='o', label="Pion")
plt.plot(energies, elecpolluted_total, marker='o', label="Electron")
plt.xlabel("Energy")
plt.ylabel("Clusters with Punch Through")
plt.title("Clusters Containing Punch Through Hits")
plt.grid(True)
plt.legend()
plt.savefig("punchclusters_both.pdf")
plt.close()

# Cluster ratios
plt.figure()
plt.plot(energies, pion_ratio_clusters, marker='o', label="Pion")
plt.plot(energies, elec_ratio_clusters, marker='o', label="Electron")
plt.xlabel("Energy")
plt.ylabel("Polluted Clusters / Total Clusters")
plt.title("Cluster Punch Through Ratio")
plt.grid(True)
plt.legend()
plt.savefig("ratioclusters_both.pdf")
plt.close()

#Number polluted events
plt.figure()
plt.plot(energies, pionpolluted_events, marker='o', label="Pion")
plt.plot(energies, elecpolluted_events, marker='o', label="Electron")
plt.xlabel("Energy")
plt.ylabel("Number of Polluted events out of 10000")
plt.title("Event Punch through Count")
plt.grid(True)
plt.legend()
plt.savefig("ratioevents_both.pdf")
plt.close()



'''
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
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec", "EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
elecweird = []
elecnormal = []
elecclusters_total= []
elecpolluted_total = []
energies = [1, 2, 5, 10, 50, 100, 150, 200]
for num in energies:
    count_weird = 0
    count_total = 0
    clusters = 0
    polluted = 0
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    # Load cluster–hit mapping arrays
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
    # Preload calorimeter arrays
    pos = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        pos[name] = {
            "x": events[f"{prefix}.position.x"].array(),
            "y": events[f"{prefix}.position.y"].array(),
            "z": events[f"{prefix}.position.z"].array(),
        }

    # Loop over events
    for i in range(events.num_entries):
        if i % 100 == 0:
            print(f"  Event {i}")

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]
        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]
        # Loop over clusters in this event
        for j in range(len(hits_begin_arr)):
            polute_boolean = False
            clusters +=1
            start = hits_begin_arr[j]
            end = hits_end_arr[j]
            # vectorized slices of hits
            indices = hit_index[start:end]
            ids = collection_ID[start:end]
            count_total +=len(ids)
            if len(indices) == 0:
                continue
            for sysid in ids:
                sysname = system2name[sysid]
                if sysname == "Skip":
                    count_weird +=1
                    polute_boolean = True
            if polute_boolean == True:
                polluted +=1
    clusters_total.append(clusters)
    polluted_total.append(polluted)
    weird.append(count_weird)
    normal.append(count_total)
elecweird = np.array(weird)
plt.figure(figsize=(10,5))
plt.plot(energies, weird, marker='o')
plt.xlabel("Energy")
plt.ylabel("Punch Through Count")
plt.title("Electrons Punch Through Counts")
plt.grid(True)
plt.savefig(f"weirdhits_electrons.pdf")
plt.close()
elecnormal = np.array(normal)
ratio = np.array(elecweird/elecnormal)
elecpolluted_total = np.array(polluted_total)
elecclusters_total = np.array(clusters_total)
elecratio_clusters = np.array(elecpolluted_total/elecclusters_total)




#Now we do pions
pionweird = []
pionnormal = []
pionclusters_total= []
pionpolluted_total = []
energies = [1, 2, 5, 10, 50, 100, 150, 200]
for num in energies:
    count_weird = 0
    count_total = 0
    clusters = 0
    polluted = 0
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    # Load cluster–hit mapping arrays
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    pandora_clusters = events["PandoraClusters"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
    # Preload calorimeter arrays
    pos = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        pos[name] = {
            "x": events[f"{prefix}.position.x"].array(),
            "y": events[f"{prefix}.position.y"].array(),
            "z": events[f"{prefix}.position.z"].array(),
        }

    # Loop over events
    for i in range(events.num_entries):
        if i % 100 == 0:
            print(f"  Event {i}")

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]
        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]
        # Loop over clusters in this event
        for j in range(len(hits_begin_arr)):
            polute_boolean = False
            clusters +=1
            start = hits_begin_arr[j]
            end = hits_end_arr[j]
            # vectorized slices of hits
            indices = hit_index[start:end]
            ids = collection_ID[start:end]
            count_total +=len(ids)
            if len(indices) == 0:
                continue
            for sysid in ids:
                sysname = system2name[sysid]
                if sysname == "Skip":
                    count_weird +=1
                    polute_boolean = True
            if polute_boolean == True:
                polluted +=1
    clusters_total.append(clusters)
    polluted_total.append(polluted)
    weird.append(count_weird)
    normal.append(count_total)
pionweird = np.array(weird)
plt.figure(figsize=(10,5))
plt.plot(energies, weird, marker='o')
plt.xlabel("Energy")
plt.ylabel("Punch Through Count")
plt.title("Pion Punch Through Count")
plt.grid(True)
plt.savefig(f"weirdhits_pions.pdf")
plt.close()
pionnormal = np.array(normal)
ratio = np.array(pionweird/pionnormal)
pionpolluted_total = np.array(polluted_total)
pionclusters_total = np.array(clusters_total)
pionratio_clusters = np.array(pionpolluted_total/pionclusters_total)



#Now we are going to graph both on one graph 
#Graph punch through hits
plt.plot(energies, pionwierd, marker = 'o', label="Pion")
plt.plot(energies, elecweird, marker = 'o', label='Electron')
plt.xlabel("Energy")
plt.ylabel("Punch Through Counts")
plt.title("Punch Through Hits for Electrons & Pions")
plt.grid(True)
plt.legend()
plt.savefig(f"punchhits_both.pdf")
plt.close()

#Now we will plot ratio
plt.plot(energies, pionratio, marker = 'o', label="Pion")
plt.plot(energies, elecratio, marker = 'o', label='Electron')
plt.xlabel("Energy")
plt.ylabel("Punch Through Counts / Total Counts")
plt.title("Punch Through Hits / Total hits for Electrons & Pions")
plt.grid(True)
plt.legend()
plt.savefig(f"ratiohits_both.pdf")
plt.close()

#Now we will just look at infected clusters

plt.plot(energies, pionpollutedtotal, marker = 'o', label="Pion")
plt.plot(energies, elecpollutedtotal, marker = 'o', label='Electron')
plt.xlabel("Energy")
plt.ylabel("Cluster Counts with Punch Through")
plt.title("Punch Through Clusters for Electrons & Pions")
plt.grid(True)
plt.legend()
plt.savefig(f"punchclusters_both.pdf")
plt.close()

plt.plot(energies, pion_ratio_clusters, marker = 'o', label="Pion")
plt.plot(energies, elec_ratio_clusters, marker = 'o', label='Electron')
plt.xlabel("Energy")
plt.ylabel("Cluster Counts with Punch Through / Total Cluster Count")
plt.title("Ratio of Clusters with Punch Through for Electrons and Pions")
plt.grid(True)
plt.legend()
plt.savefig(f"ratioclusters_both.pdf")
plt.close()


'''

