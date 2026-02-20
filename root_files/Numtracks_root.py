#Number of Tracks and Corresponding Clusters

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

#choices = [1, 2, 5, 10, 50, 100, 150, 200]
choices = [2, 50]
#First I will see the ratio of particle w/ status 0, its energy to leading cluster energy
bib_mean = []
bib_low = []
bib_high = []
nobib_mean = []
nobib_low = []
nobib_high = []

for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    all_tracks = events["AllTracks"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    index_deduped = deduped_tracks["DedupedTracks_objIdx.index"].array()
    track_states = events["_AllTracks_trackStates"]
    cluster_energy = clusters["PandoraClusters.energy"].array()
    print(f"Processing {num} GeV")
    number_tracks = []
    for i in range(events.num_entries):
        indexes = index_deduped[i]
        cluster_energies = cluster_energy[i]
        num_clusters = len(cluster_energies)
        num_tracks = len(indexes)
        number_tracks.append(num_tracks)
        if (num_tracks !=1):
            print("Bib")
            print(i)
            print(f"tracks: {num_tracks}")
            print(f"clusters: {num_clusters}")
    num_tracks = np.array(number_tracks)
    bib_mean.append(np.median(num_tracks))
    median = np.median(num_tracks)
    q16, q84 = np.percentile(num_tracks, [16, 84])
    bib_low.append(median - q16)
    bib_high.append(q84 - median)
    bins = np.arange(np.min(num_tracks), np.max(num_tracks) + 2) - 0.5
    plt.hist(num_tracks, bins=bins, edgecolor='black')
    #Adding log scale
    plt.yscale("log")
    plt.xlabel("Number of Tracks per Event ")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Track Count {num} GeV Pions with Bib")
    plt.tight_layout()
    plt.savefig(f"num_tracks_pions_bib{num}GeV.pdf")
    plt.close()

for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    all_tracks = events["AllTracks"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    index_deduped = deduped_tracks["DedupedTracks_objIdx.index"].array()
    track_states = events["_AllTracks_trackStates"]
    cluster_energy = clusters["PandoraClusters.energy"].array()
    print(f"Processing {num} GeV")
    number_tracks = []
    for i in range(events.num_entries):
        indexes = index_deduped[i]
        cluster_energies = cluster_energy[i]
        num_clusters = len(cluster_energies)
        num_tracks = len(indexes)
        number_tracks.append(num_tracks)
        if (num_tracks !=1):
            print("Pion")
            print(i)
            print(f"tracks: {num_tracks}")
            print(f"clusters: {num_clusters}")
    num_tracks = np.array(number_tracks)
    nobib_mean.append(np.median(num_tracks))
    median = np.median(num_tracks)
    q16, q84 = np.percentile(num_tracks, [16, 84])
    nobib_low.append(median - q16)
    nobib_high.append(q84 - median)
    bins = np.arange(np.min(num_tracks), np.max(num_tracks) + 2) - 0.5
    plt.hist(num_tracks, bins=bins, edgecolor='black')
    plt.xlabel("Number of Tracks per Event")
    plt.ylabel("Count")
    #adding log scale
    plt.yscale("log")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Track Count {num} GeV Pions No Bib")
    plt.tight_layout()
    plt.savefig(f"num_tracks_pions_nobib{num}GeV.pdf")
    plt.close()

plt.errorbar(choices, bib_mean, yerr=[bib_low, bib_high], fmt='s', alpha= 0.6, capsize=4, label="Pions with Bib")
plt.errorbar(choices, nobib_mean, yerr=[nobib_low, nobib_high], fmt='o', alpha=0.6, capsize=4, label="Pions without Bib")
plt.xlabel("Beam Energy")
plt.ylabel("Median Number of Tracks per Event")
plt.title("Median Number of Tracks versus beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_tracks_bib.pdf")
plt.close()

#'/users/rldohert/data/mucoll/rldohert/pdg_211_pt_2_theta_15-15/reco_pdg_211_pt_2_theta_15-15_nobib.root