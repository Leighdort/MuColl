#Difference_phi_root.py
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
choices = [1, 2]
#First I will see the ratio of particle w/ status 0, its energy to leading cluster energy
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_-11_pt_{num}_theta_15-15/reco_pdg_-11_pt_{num}_theta_15-15.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    all_tracks = events["AllTracks"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    index_deduped = deduped_tracks["DedupedTracks_objIdx.index"].array()
    track_states = events["_AllTracks_trackStates"]
    tan_lambda = track_states["_AllTracks_trackStates.tanLambda"].array()
    phi_tracks = track_states["_AllTracks_trackStates.phi"].array()
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
        t_phi = phi_tracks[i]
        indices = index_deduped[i]
        if len(indices) == 0:
            print("cat")
            continue
        if len(indices) !=1:
            print("meow")
            continue
        meaningful_track = indices[0]
        our_tphi = t_phi[meaningful_track]
        track_degrees = np.degrees(our_tphi)
        cluster_phi = np.arctan2(cluster_y[i], cluster_x[i])
        cluster_phi = np.degrees(cluster_phi)
        leading_index = np.argmax(cluster_energy[i])
        cluster_phi = cluster_phi[leading_index]
        difference_phi = track_degrees - cluster_phi
        difference_phi = (difference_phi + 180) % 360 - 180
        if not np.isscalar(difference_phi):
            print("\n🚨 Non-scalar difference detected!")
            print(f"Energy = {num}   Event = {i}")
            print(f"difference = {difference_phi}")
            print(f"track_degrees = {track_degrees}")
            print(f"cluster_phi = {cluster_phi}")
            print("---")
            continue
        distance_for_this_energy.append(difference_phi)
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
    plt.title(f"Track-cluster (theta) {num} energy Positrons")
    plt.tight_layout()
    plt.savefig(f"difference_positrons_phi{num}GeV.pdf")
    plt.close()


