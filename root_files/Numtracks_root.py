#Number of Tracks and Corresponding Clusters

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

choices = [1, 2, 5, 10, 50, 100, 150, 200]
#First I will see the ratio of particle w/ status 0, its energy to leading cluster energy
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
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
            print(f"tracks: {num_tracks}")
            print(f"clusters: {num_clusters}")
    num_tracks = np.array(number_tracks)
    bins = np.arange(np.min(num_tracks), np.max(num_tracks) + 2) - 0.5
    plt.hist(num_tracks, bins=bins, edgecolor='black')
    plt.xlabel("Track Count")
    plt.ylabel("Count")
    plt.title(f"Track Count {num} energy pions")
    plt.tight_layout()
    plt.savefig(f"num_tracks_pions{num}GeV.pdf")
    plt.close()