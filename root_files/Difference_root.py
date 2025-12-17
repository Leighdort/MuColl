#Now let's look at tracks
#Deduped tracks gives the index of All_trcaks you want


#Comparing energy to total energy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

electron_mean = []
electron_low = []
electron_high = []
pion_mean = []
pion_low = []
pion_high = []
choices = [1, 2, 5, 10, 50, 100, 150, 200]
#First I will see the ratio of particle w/ status 0, its energy to leading cluster energy
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    all_tracks = events["AllTracks"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    index_deduped = deduped_tracks["DedupedTracks_objIdx.index"].array()
    track_states = events["_AllTracks_trackStates"]
    tan_lambda = track_states["_AllTracks_trackStates.tanLambda"].array()
    cluster_x = clusters["PandoraClusters.position.x"].array()
    cluster_y = clusters["PandoraClusters.position.y"].array()
    cluster_z = clusters["PandoraClusters.position.z"].array()
    cluster_energy = clusters["PandoraClusters.energy"].array()
    print(f"Processing {num} GeV")
    distance_for_this_energy = []
    for i in range(events.num_entries):
        x_cluster = cluster_x[i]
        y_cluster = cluster_y[i]
        z_cluster = cluster_z[i]
        t_lambda = tan_lambda[i]
        indices = index_deduped[i]
        if len(indices) == 0:
            print("cat")
            continue
        if len(indices) !=1:
            print("meow")
            continue
        meaningful_track = indices[0]
        our_tlambda = t_lambda[meaningful_track]
        track = np.arctan(our_tlambda)
        track_theta = (np.pi / 2) - track
        track_degrees = np.degrees(track_theta)
        pz = cluster_z[i]
        pt = np.sqrt(cluster_x[i]**2 + cluster_y[i]**2)
        cluster_theta = np.arctan2(pt, pz)
        cluster_degrees = np.degrees(cluster_theta)
        leading_index = np.argmax(cluster_energy[i])
        cluster_degrees = cluster_degrees[leading_index]
        difference = track_degrees - cluster_degrees
        if not np.isscalar(difference):
            print("\n🚨 Non-scalar difference detected!")
            print(f"Energy = {num}   Event = {i}")
            print(f"difference = {difference}")
            print(f"track_degrees = {track_degrees}")
            print(f"cluster_degrees = {cluster_degrees}")
            print("---")
            continue
        distance_for_this_energy.append(difference)
    distance_for_this = np.array(distance_for_this_energy)
    electron_mean.append(np.median(distance_for_this))
    #electron_std.append(np.std(distance_for_this))
    mean = np.median(distance_for_this)
    q16, q84 = np.percentile(distance_for_this, [16, 84])
    electron_low.append(mean-q16)
    electron_high.append(q84-mean)
    bins = np.linspace(np.min(distance_for_this), np.max(distance_for_this), 30)
    plt.hist(distance_for_this, bins=bins, edgecolor='black')
    plt.axvline(mean,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {mean:.2f}")
    plt.legend()
    plt.xlabel("Track theta - leading cluster theta")
    plt.ylabel("Count")
    plt.title(f"Track-cluster (theta) {num} energy Electrons")
    plt.tight_layout()
    plt.savefig(f"difference_electrons{num}GeV.pdf")
    plt.close()



    #Now we will do pions: 

for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    all_tracks = events["AllTracks"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    index_deduped = deduped_tracks["DedupedTracks_objIdx.index"].array()
    track_states = events["_AllTracks_trackStates"]
    tan_lambda = track_states["_AllTracks_trackStates.tanLambda"].array()
    cluster_x = clusters["PandoraClusters.position.x"].array()
    cluster_y = clusters["PandoraClusters.position.y"].array()
    cluster_z = clusters["PandoraClusters.position.z"].array()
    cluster_energy = clusters["PandoraClusters.energy"].array()
    print(f"Processing {num} GeV")
    distance_for_this_energy = []
    for i in range(events.num_entries):
        x_cluster = cluster_x[i]
        y_cluster = cluster_y[i]
        z_cluster = cluster_z[i]
        t_lambda = tan_lambda[i]
        indices = index_deduped[i]
        if len(indices) == 0:
            print("cat")
            continue
        if len(indices) !=1:
            print("meow")
            continue
        meaningful_track = indices[0]
        our_tlambda = t_lambda[meaningful_track]
        track = np.arctan(our_tlambda)
        track_theta = (np.pi / 2) - track
        track_degrees = np.degrees(track_theta)
        pz = cluster_z[i]
        pt = np.sqrt(cluster_x[i]**2 + cluster_y[i]**2)
        cluster_theta = np.arctan2(pt, pz)
        cluster_degrees = np.degrees(cluster_theta)
        leading_index = np.argmax(cluster_energy[i])
        cluster_degrees = cluster_degrees[leading_index]
        difference = track_degrees - cluster_degrees
        if not np.isscalar(difference):
            print("\n🚨 Non-scalar difference detected!")
            print(f"Energy = {num}   Event = {i}")
            print(f"difference = {difference}")
            print(f"track_degrees = {track_degrees}")
            print(f"cluster_degrees = {cluster_degrees}")
            print("---")
            continue
        distance_for_this_energy.append(difference)
    distance_for_this = np.array(distance_for_this_energy)
    pion_mean.append(np.median(distance_for_this))
    mean = np.median(distance_for_this)
    q16, q84 = np.percentile(distance_for_this, [16, 84])
    pion_low.append(mean-q16)
    pion_high.append(q84-mean)
    bins = np.linspace(np.min(distance_for_this), np.max(distance_for_this), 30)
    plt.hist(distance_for_this, bins=bins, edgecolor='black')
    plt.axvline(mean,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {mean:.2f}")
    plt.legend()
    plt.xlabel("Track theta - leading cluster theta")
    plt.ylabel("Count")
    plt.title(f"Track-cluster (theta) {num} energy pions")
    plt.tight_layout()
    plt.savefig(f"difference_pions{num}GeV.pdf")
    plt.close()

plt.errorbar(choices, electron_mean, yerr=[electron_low, electron_high], fmt='o', capsize=4, label="Electrons")
plt.errorbar(choices, pion_mean, yerr=[pion_low, pion_high], fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy")
plt.ylabel("Distance between Track and Cluster ")
plt.title("Track Theta and Cluster Theta Theta")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_difference.pdf")
plt.close()
