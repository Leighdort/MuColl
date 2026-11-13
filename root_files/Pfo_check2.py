#This is checking the PFOS reconstruction energy

#We are now going to do fraction w/ all of the clusters
#Comparing energy to total energy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
choices = [50]
from matplotlib.colors import LogNorm
pfo_energy_over_track_pt = []
pfo_pt_over_track_pt = []
cluster_energy_over_track_pt = []

track_pt_values = []
track_p_values = []
pfo_energy_values = []
pfo_pt_values = []
cluster_energy_values = []
delta_phi_values = []
delta_theta_values = []
pfo_energy_over_track_p = []
cluster_energy_over_track_p = []
tracks_per_event = []
pfos_per_event = []
associated_tracks_per_event = []
associated_fraction_per_event = []
# Cluster association validation
associated_cluster_delta_r_values = []
nearest_cluster_delta_r_values = []
associated_cluster_is_nearest = []
closest_cluster_energy_over_track_p = []
associated_cluster_energy_over_track_p = []
debug_event_idx = []
debug_pfo_idx = []
debug_assoc_idx = []
debug_nearest_idx = []
sitracks_per_event = []
pfos_with_track_no_cluster = 0
def get_calo_face_state(event_i, track_index, location_target=4):
    sb = pandora_track_statesbegin[event_i][track_index]
    se = pandora_track_statesend[event_i][track_index]
    for s in range(sb, se):
        if track_states_location[event_i][s] == location_target:
            return s
    return None  # no calo-face state for this track — worth flagging, not silently

for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root")
    #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_85-85_fedecheck/job_0/reco_output_p__nobib_oldalg.edm4hep.root")
    '''
    meta_tree = file["podio_metadata"]
    meta = meta_tree["events___CollectionTypeInfo"]

    ids   = meta["events___CollectionTypeInfo.collectionID"].array()[0]
    names = meta["events___CollectionTypeInfo.name"].array()[0]
    types = meta["events___CollectionTypeInfo.dataType"].array()[0]

    collid2name = {int(i): str(n) for i, n in zip(ids, names)}

    for i, n, t in zip(ids, names, types):
        print(i, n, t)
    ''' 
    events = file["events"]
    clusters = events["PandoraClusters"]
    mcparticles = events["MCParticles"]
    sitracks = events["SiTracks"]
    cluster_energies = clusters["PandoraClusters.energy"].array()
    pdg_momentum_x = mcparticles["MCParticles.momentum.x"].array()
    pdg_momentum_y = mcparticles["MCParticles.momentum.y"].array()
    pdg_momentum_z = mcparticles["MCParticles.momentum.z"].array()
    statuses = mcparticles["MCParticles.generatorStatus"].array()
    masses = mcparticles["MCParticles.mass"].array()
    pfo_energy = events["PandoraPFOs/PandoraPFOs.energy"].array()
    pandora_tracks_begin = events["PandoraPFOs/PandoraPFOs.tracks_begin"].array()
    pandora_tracks_end   = events["PandoraPFOs/PandoraPFOs.tracks_end"].array()
    pandora_clusters_begin = events["PandoraPFOs/PandoraPFOs.clusters_begin"].array()
    pandora_clusters_end   = events["PandoraPFOs/PandoraPFOs.clusters_end"].array()
    pandora_clusters_index = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"].array()
    pandora_clusters_collection = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.collectionID"].array()
    pandora_tracks_index = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"].array()
    pandora_tracks_collection = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.collectionID"].array()
    pandora_track_statesbegin = events["AllTracks/AllTracks.trackStates_begin"].array()
    pandora_track_statesend = events["AllTracks/AllTracks.trackStates_end"].array()
    cluster_energies = events["PandoraClusters/PandoraClusters.energy"].array()  # adjust name as needed
    track_states_phi = events["_AllTracks_trackStates/_AllTracks_trackStates.phi"].array()
    track_states_omega = events[ "_AllTracks_trackStates/_AllTracks_trackStates.omega" ].array()
    track_states_tanLambda = events[ "_AllTracks_trackStates/_AllTracks_trackStates.tanLambda" ].array()
    pfo_momx = events["PandoraPFOs/PandoraPFOs.momentum.x"].array()
    pfo_momy = events["PandoraPFOs/PandoraPFOs.momentum.y"].array()
    pfo_momz = events["PandoraPFOs/PandoraPFOs.momentum.z"].array()
    cluster_x = events["PandoraClusters/PandoraClusters.position.x"].array()
    cluster_y = events["PandoraClusters/PandoraClusters.position.y"].array()
    cluster_z = events["PandoraClusters/PandoraClusters.position.z"].array()
    cluster_type = events["PandoraClusters/PandoraClusters.type"].array()
    track_states_x = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.x"].array()
    track_states_y = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.y"].array()
    track_states_z = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.z"].array()
    track_states_location = events["_AllTracks_trackStates/_AllTracks_trackStates.location"].array()
    B = 5
    #for i in range(100):

    l = 0
    for pfo in range(len(pfo_energy[l])):
        tb = pandora_tracks_begin[l][pfo]
        te = pandora_tracks_end[l][pfo]
        for j in range(tb, te):
            if pandora_tracks_collection[l][j] != 4111979324:
                continue
            track_index = pandora_tracks_index[l][j]
            sb = pandora_track_statesbegin[l][track_index]
            se = pandora_track_statesend[l][track_index]
            print(f"Track {track_index} has {se-sb} states:")
            for s in range(sb, se):
                x, y, z = track_states_x[l][s], track_states_y[l][s], track_states_z[l][s]
                r = np.sqrt(x**2 + y**2)
                print(f"  state {s}: location={track_states_location[l][s]}, "
                    f"r={r:.1f} mm, z={z:.1f} mm, x={x:.2f}, y={y:.2f}")
    for i in range(events.num_entries):
        print(i)
        # Loop over PFOs
        num_total_tracks = len(pandora_track_statesbegin[i])
        num_total_pfos = len(pfo_energy[i])
        num_associated_tracks = 0
        associated_track_indices = set()
        
        for pfo in range(len(pfo_energy[i])):
            print(f"\n--- PFO {pfo} ---")

            # Track relations belonging to this PFO
            track_begin = pandora_tracks_begin[i][pfo]
            track_end = pandora_tracks_end[i][pfo]

            print(f"Track relations: {track_begin} -> {track_end}")

            has_track = False
            pT_track = None

            # ---------------------------------------------------------
            # Look for a track associated with this PFO
            # ---------------------------------------------------------
            for j in range(track_begin, track_end):

                track_index = pandora_tracks_index[i][j]
                collection_id = pandora_tracks_collection[i][j]

                # Only look at AllTracks
                if collection_id == 4111979324:
                    associated_track_indices.add(track_index)
                    has_track = True
                    state_begin = pandora_track_statesbegin[i][track_index]
                    omega_track = track_states_omega[i][state_begin]
                    phi_track = track_states_phi[i][state_begin]
                    tanlambda_track = track_states_tanLambda[i][state_begin]
                    theta_track = np.arctan2(1.0, tanlambda_track)
                    
                    calo_state = get_calo_face_state(i, track_index)
                    if calo_state is None:
                        print(f"WARNING: event {i} pfo {pfo} track {track_index} has no AtCalorimeter state — skipping")
                        has_track = False
                        continue
                    if calo_state is not None:
                        x = track_states_x[i][calo_state]
                        y = track_states_y[i][calo_state]
                        z = track_states_z[i][calo_state]
                        phi_spatial_track = np.arctan2(y, x)
                        theta_spatial_track = np.arctan2(np.sqrt(x**2 + y**2), z)

                        '''
                        print(f"phi_spatial {phi_spatial_track}")
                        print(f"theta_spatial {theta_spatial_track}")
                        print(f"phi {phi_track}")
                        print(f"theta {theta_track}")
                        '''
                    # Track pT in GeV
                    pT_track = abs(0.3 * B / omega_track / 1000)
                    p_track = pT_track / np.sin(theta_spatial_track)
                    break
            
            # ---------------------------------------------------------
            # Only continue if this PFO has a track
            # ---------------------------------------------------------
            if not has_track:
                continue
            
            # ---------------------------------------------------------
            # Get cluster energy associated with this PFO
            # ---------------------------------------------------------
            cluster_begin = pandora_clusters_begin[i][pfo]
            cluster_end = pandora_clusters_end[i][pfo]
            n_clusters_this_pfo = cluster_end - cluster_begin
            if n_clusters_this_pfo != 1:
                print(f"FOUND IT: event {i} pfo {pfo} has {n_clusters_this_pfo} clusters (not 1)")

            if cluster_begin == cluster_end:
                print(f"WARNING: event {i} pfo {pfo} has a track but NO cluster — skipping")
                pfos_with_track_no_cluster += 1
                continue
            pfo_cluster_energy = 0.0

            for j in range(cluster_begin, cluster_end):

                cluster_index = pandora_clusters_index[i][j]

                pfo_cluster_energy += cluster_energies[i][cluster_index]
                cluster_x_pos = cluster_x[i][cluster_index]
                cluster_y_pos = cluster_y[i][cluster_index]
                cluster_z_pos = cluster_z[i][cluster_index]
                cluster_phi = np.arctan2(cluster_y_pos, cluster_x_pos)
                cluster_theta = np.arctan2(np.sqrt(cluster_x_pos**2 + cluster_y_pos**2), cluster_z_pos)
                        # ---------------------------------------------------------
            # Cluster association validation
            #
            # Compare the cluster Pandora associated with this PFO
            # to the nearest cluster in the entire event.
            # ---------------------------------------------------------

            # There is exactly one cluster per PFO
            associated_cluster_index = pandora_clusters_index[i][cluster_begin]

            # Direction of the associated cluster
            associated_cluster_phi = np.arctan2(
                cluster_y[ i ][associated_cluster_index],
                cluster_x[ i ][associated_cluster_index]
            )

            associated_cluster_theta = np.arctan2(
                np.sqrt(
                    cluster_x[i][associated_cluster_index]**2
                    + cluster_y[i][associated_cluster_index]**2
                ),
                cluster_z[i][associated_cluster_index]
            )

            # Angular separation between track and associated cluster
            associated_delta_phi = np.arctan2(
                np.sin(associated_cluster_phi - phi_spatial_track),
                np.cos(associated_cluster_phi - phi_spatial_track)
            )

            associated_delta_theta = (
                associated_cluster_theta - theta_spatial_track
            )

            associated_delta_r = np.sqrt(
                associated_delta_phi**2
                + associated_delta_theta**2
            )

            # ---------------------------------------------------------
            # Find the nearest cluster to the track among ALL clusters
            # in this event
            # ---------------------------------------------------------

            nearest_delta_r = np.inf
            nearest_cluster_index = None
            closest_cluster_energy = 0.0

            for cluster_index_all in range(len(cluster_energies[i])):

                x = cluster_x[i][cluster_index_all]
                y = cluster_y[i][cluster_index_all]
                z = cluster_z[i][cluster_index_all]

                cluster_phi_all = np.arctan2(y, x)

                cluster_theta_all = np.arctan2(
                    np.sqrt(x**2 + y**2),
                    z
                )

                delta_phi_all = np.arctan2(
                    np.sin(cluster_phi_all - phi_spatial_track),
                    np.cos(cluster_phi_all - phi_spatial_track)
                )

                delta_theta_all = (
                    cluster_theta_all - theta_spatial_track
                )

                delta_r_all = np.sqrt(
                    delta_phi_all**2
                    + delta_theta_all**2
                )

                
                if delta_r_all < nearest_delta_r:
                    nearest_delta_r = delta_r_all
                    nearest_cluster_index = cluster_index_all
                    closest_cluster_energy = cluster_energies[i][cluster_index_all]

            # ---------------------------------------------------------
            # Store results
            # ---------------------------------------------------------

            associated_cluster_delta_r_values.append(
                associated_delta_r
            )
            associated_cluster_energy_over_track_p.append(
                pfo_cluster_energy / p_track
            )
            nearest_cluster_delta_r_values.append(
                nearest_delta_r
            )

            closest_cluster_energy_over_track_p.append(
                closest_cluster_energy / p_track
            )

            is_nearest = (
                associated_cluster_index == nearest_cluster_index
            )

            associated_cluster_is_nearest.append(is_nearest)
            debug_event_idx.append(i)
            debug_pfo_idx.append(pfo)
            debug_assoc_idx.append(associated_cluster_index)
            debug_nearest_idx.append(nearest_cluster_index)
            '''
            print(
                f"    associated cluster deltaR = "
                f"{associated_delta_r:.5f}"
            )

            print(
                f"    nearest cluster deltaR = "
                f"{nearest_delta_r:.5f}"
            )

            print(
                f"    Pandora cluster is nearest = "
                f"{is_nearest}"
            )
            '''
            # ---------------------------------------------------------
            # PFO quantities
            # ---------------------------------------------------------
            px_pfo = pfo_momx[i][pfo]
            py_pfo = pfo_momy[i][pfo]

            pT_pfo = np.sqrt(px_pfo**2 + py_pfo**2)
            E_pfo = pfo_energy[i][pfo]
            '''
            print(f"    track pT = {pT_track:.4f} GeV")
            print(f"    PFO pT   = {pT_pfo:.4f} GeV")
            print(f"    PFO E    = {E_pfo:.4f} GeV")
            print(f"    cluster E = {pfo_cluster_energy:.4f} GeV")
            '''
            # ---------------------------------------------------------
            # Validation quantities
            # ---------------------------------------------------------
            delta_phi = associated_delta_phi
            delta_theta = associated_delta_theta
            '''
            delta_phi = np.arctan2(
                np.sin(cluster_phi - phi_track),
                np.cos(cluster_phi - phi_track)
            )
            delta_theta = cluster_theta - theta_track
            delta_phi_values.append(delta_phi)
            '''
            delta_theta_values.append(delta_theta)
            delta_phi_values.append(delta_phi)
            if pT_track > 0:

                pfo_energy_over_track_p.append(
                    E_pfo / p_track
                )

                cluster_energy_over_track_p.append(
                    pfo_cluster_energy / p_track
                )
                pfo_energy_over_track_pt.append(
                    E_pfo / pT_track
                )

                pfo_pt_over_track_pt.append(
                    pT_pfo / pT_track
                )

                cluster_energy_over_track_pt.append(
                    pfo_cluster_energy / pT_track
                )
                # Keep individual values too
                track_pt_values.append(pT_track)
                track_p_values.append(p_track)
                pfo_energy_values.append(E_pfo)
                pfo_pt_values.append(pT_pfo)
                cluster_energy_values.append(pfo_cluster_energy)
                '''
                print(f"    E_PFO / pT_track = {E_pfo / pT_track:.4f}")
                print(f"    pT_PFO / pT_track = {pT_pfo / pT_track:.4f}")
                print(
                    f"    E_cluster / pT_track = "
                    f"{pfo_cluster_energy / pT_track:.4f}"
                )
                '''

            # Don't need to look at another AllTracks relation
            # if the PFO should have only one track.
            
        tracks_per_event.append(num_total_tracks)
        pfos_per_event.append(num_total_pfos)
        num_si_tracks = len(sitracks[i])
        sitracks_per_event.append(num_si_tracks)

        num_associated_tracks = len(associated_track_indices)
        associated_tracks_per_event.append(num_associated_tracks)
        association_fraction = (
            num_associated_tracks / num_total_tracks
            if num_total_tracks > 0 else 0
        )
        associated_fraction_per_event.append(association_fraction)
# ---------------------------------------------------------
# Plot 1: PFO energy / track pT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    pfo_energy_over_track_pt,
    bins=50,
    histtype="step",
)

plt.xlabel(r"$E_{\mathrm{PFO}} / p_T^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("PFO Energy / Track $p_T$")
plt.grid(alpha=0.3)

plt.savefig("pfo_energy_over_track_pt_211_85.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 2: PFO pT / track pT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    pfo_pt_over_track_pt,
    bins=50,
    histtype="step",
)

plt.xlabel(r"$p_T^{\mathrm{PFO}} / p_T^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("PFO $p_T$ / Track $p_T$")
plt.grid(alpha=0.3)

plt.savefig("pfo_pt_over_track_pt_211_85.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 3: Cluster energy / track pT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    cluster_energy_over_track_pt,
    bins=50,
    histtype="step",
)

plt.xlabel(r"$E_{\mathrm{cluster}} / p_T^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("Cluster Energy / Track $p_T$")
plt.grid(alpha=0.3)

plt.savefig("cluster_energy_over_track_pt_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 4: Cluster phi - Track phi
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    delta_phi_values,
    bins=50,
    histtype="step",
)

plt.xlabel(r"$\Delta\phi = \phi_{\mathrm{cluster}} - \phi_{\mathrm{track}}$ [rad]")
plt.ylabel("Number of charged PFOs")
plt.title(r"Cluster--Track $\Delta\phi$")
plt.grid(alpha=0.3)

plt.savefig("cluster_track_delta_phi_211_85.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 5: Cluster theta - Track theta
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    delta_theta_values,
    bins=50,
    histtype="step",
)

plt.xlabel(r"$\Delta\theta = \theta_{\mathrm{cluster}} - \theta_{\mathrm{track}}$ [rad]")
plt.ylabel("Number of charged PFOs")
plt.title(r"Cluster--Track $\Delta\theta$")
plt.grid(alpha=0.3)

plt.savefig("cluster_track_delta_theta_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 6: Delta phi vs Delta theta
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist2d(
    delta_phi_values,
    delta_theta_values,
    bins=50,
    norm=LogNorm()
)

plt.xlabel(r"$\Delta\phi$ [rad]")
plt.ylabel(r"$\Delta\theta$ [rad]")
plt.title(r"Cluster--Track Angular Differences")
plt.colorbar(label="Number of charged PFOs")

plt.savefig("cluster_track_delta_phi_vs_delta_theta_211_85.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 7: PFO energy / track p
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    pfo_energy_over_track_p,
    bins=50,
    histtype="step",
)

plt.xlabel(r"$E_{\mathrm{PFO}} / p^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("PFO Energy / Track Momentum")
plt.grid(alpha=0.3)

plt.savefig("pfo_energy_over_track_p_211_85.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 8: Cluster energy / track p
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    cluster_energy_over_track_p,
    bins=50,
    histtype="step",
)

plt.xlabel(r"$E_{\mathrm{cluster}} / p^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("Cluster Energy / Track Momentum")
plt.grid(alpha=0.3)

plt.savefig("cluster_energy_over_track_p_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 9: Number of associated tracks vs PFO multiplicity
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist2d(
    pfos_per_event,
    associated_tracks_per_event,
    bins=[50, 50],
    norm=LogNorm()
)

plt.xlabel("PFO multiplicity")
plt.ylabel("Number of tracks associated with PFOs")
plt.title("Associated Tracks vs PFO Multiplicity")
plt.colorbar(label="Number of events")

plt.savefig("associated_tracks_vs_pfo_multiplicity_211_85.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 10: Number of associated tracks vs track multiplicity
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist2d(
    tracks_per_event,
    associated_tracks_per_event,
    bins=[50, 50],
    norm=LogNorm()
)

plt.xlabel("AllTrack multiplicity ")
plt.ylabel("Number of tracks associated with PFOs")
plt.title("Associated Tracks vs AllTrack Multiplicity")
plt.colorbar(label="Number of events")

plt.savefig("associated_tracks_vs_track_multiplicity_211_85.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot: Track-PFO association fraction per event
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    associated_fraction_per_event,
    bins=50,
    histtype="step",
  
)

plt.xlabel(r"$N_{\mathrm{tracks\ with\ a\ PFO}} / N_{\mathrm{tracks}}$")
plt.ylabel("Number of events")
plt.title("Track-to-PFO Association Efficiency")
plt.xlim(0, 1)
plt.grid(alpha=0.3)

plt.savefig(
    "track_pfo_association_fraction_per_event_211_85.png",
    dpi=150
)
plt.close()


#Cluster Energy vs Track pT
plt.figure(figsize=(8, 6))

plt.hist2d(
    track_pt_values,
    cluster_energy_values,
    bins=[50, 50],
    norm=LogNorm()
)

plt.xlabel(r"$p_T^{\mathrm{track}}$ [GeV]")
plt.ylabel(r"$E_{\mathrm{cluster}}$ [GeV]")
plt.title("Cluster Energy vs Track $p_T$")
plt.colorbar(label="Number of charged PFOs")

plt.savefig(
    "cluster_energy_vs_track_pt_211_85.png",
    dpi=150
)
plt.close()

plt.figure(figsize=(8, 6))

plt.hist2d(
    track_p_values,
    cluster_energy_values,
    bins=[50, 50],
    norm=LogNorm()
)

plt.xlabel(r"$p^{\mathrm{track}}$ [GeV]")
plt.ylabel(r"$E_{\mathrm{cluster}}$ [GeV]")
plt.title("Cluster Energy vs Track $p$")
plt.colorbar(label="Number of charged PFOs")

plt.savefig(
    "cluster_energy_vs_track_p_211_85.png",
    dpi=150
)
plt.close()

#Lastlly we will do a nearest cluster delta R versus delta R of the cluster Pandora actually associates with the PFO
#We will do N(pandora associated cluster is nearest to track) / N (tracks with a cluster)

# ---------------------------------------------------------
# Plot 11: Pandora-associated cluster deltaR vs
# nearest cluster deltaR
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist2d(
    nearest_cluster_delta_r_values,
    associated_cluster_delta_r_values,
    bins=50,
    norm=LogNorm()
)

# y = x reference line
max_delta_r = max(
    max(nearest_cluster_delta_r_values),
    max(associated_cluster_delta_r_values)
)

plt.plot(
    [0, max_delta_r],
    [0, max_delta_r],
    linestyle="--",
    label=r"$y=x$"
)

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title("Pandora Cluster Association vs Nearest Cluster")
plt.colorbar(label="Number of charged PFOs")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(
    "associated_vs_nearest_cluster_deltaR_211_85.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# Cluster association fraction
# ---------------------------------------------------------

num_tracks_with_cluster = len(associated_cluster_is_nearest)
num_associated_is_nearest = np.sum(
    associated_cluster_is_nearest
)

if num_tracks_with_cluster > 0:

    cluster_association_fraction = (
        num_associated_is_nearest
        / num_tracks_with_cluster
    )

    print("\n-----------------------------------------")
    print("Cluster Association Validation")
    print("-----------------------------------------")
    print(
        f"Tracks with a cluster: "
        f"{num_tracks_with_cluster}"
    )
    print(
        f"Pandora cluster is nearest: "
        f"{num_associated_is_nearest}"
    )
    print(
        f"Association fraction: "
        f"{cluster_association_fraction:.4f}"
    )
    print(
        f"Association fraction (%): "
        f"{100 * cluster_association_fraction:.2f}%"
    )
    print("-----------------------------------------")


# ---------------------------------------------------------
# Plot 12: Pandora-associated cluster vs nearest cluster
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    associated_cluster_is_nearest,
    bins=[-0.5, 0.5, 1.5],
    histtype="step"
)

plt.xticks(
    [0, 1],
    ["Not nearest", "Nearest"]
)

plt.xlabel(
    "Is Pandora-associated cluster the nearest cluster?"
)
plt.ylabel("Number of charged PFOs")
plt.title("Track--Cluster Association")
plt.grid(alpha=0.3)

plt.savefig(
    "pandora_cluster_association_211_85.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# Plot 13: Pandora-associated cluster deltaR vspfocheck2_5166352.err
# nearest cluster deltaR
#
# Color = E_cluster / p_track
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    nearest_cluster_delta_r_values,
    associated_cluster_delta_r_values,
    c=associated_cluster_energy_over_track_p,
    s=10,
    alpha=0.5,
    norm=LogNorm()
)

# y = x reference line
max_delta_r = max(
    max(nearest_cluster_delta_r_values),
    max(associated_cluster_delta_r_values)
)

plt.plot(
    [0, max_delta_r],
    [0, max_delta_r],
    linestyle="--",
    label=r"$y=x$"
)

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title(r"Cluster Association: $\Delta R$ Comparison")

plt.colorbar(
    scatter,
    label=r"$E_{\mathrm{cluster}} / p_{\mathrm{track}}$"
)

plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 0.45)
plt.ylim(0, 0.45)
plt.savefig(
    "associated_vs_nearest_cluster_deltaR_colored_energy_211_85.png",
    dpi=150
)

plt.close()



# ---------------------------------------------------------
# Plot 14: Pandora- cluster deltaR vs
# nearest cluster deltaR
#
# Color = E_closest_cluster / p_track
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    nearest_cluster_delta_r_values,
    associated_cluster_delta_r_values,
    c=closest_cluster_energy_over_track_p,
    s=10,
    alpha=0.5,
    norm=LogNorm()
)

# y = x reference line
max_delta_r = max(
    max(nearest_cluster_delta_r_values),
    max(associated_cluster_delta_r_values),
    
)

plt.plot(
    [0, max_delta_r],
    [0, max_delta_r],
    linestyle="--",
    label=r"$y=x$"
)

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title(r"Cluster Association: $\Delta R$ Comparison")

plt.colorbar(
    scatter,
    label=r"$E_{\mathrm{cluster}} (closest cluster) / p_{\mathrm{track}}$"
)

plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 0.45)
plt.ylim(0, 0.45)
plt.savefig(
    "associated_vs_nearest_cluster_deltaR_colored_closest_energy_211_85.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# Plot 15: Which cluster has the better energy match?
#
# Red  = Pandora-associated cluster has E/p closer to 1
# Blue = nearest cluster has E/p closer to 1
# ---------------------------------------------------------

pandora_better = []
closest_better = []
ties = []
same_clus = []

for pandora_ep, closest_ep, is_nearest in zip(
    cluster_energy_over_track_p,
    closest_cluster_energy_over_track_p,
    associated_cluster_is_nearest
):

    pandora_distance = abs(pandora_ep - 1.0)
    closest_distance = abs(closest_ep - 1.0)

    if pandora_distance < closest_distance:
        pandora_better.append(True)
        closest_better.append(False)
        ties.append(False)

    elif closest_distance < pandora_distance:
        pandora_better.append(False)
        closest_better.append(True)
        ties.append(False)

    else:
        pandora_better.append(False)
        closest_better.append(False)
        ties.append(True)

        # Among the E/p ties, was it actually the same cluster?
        same_clus.append(is_nearest)


# Convert to arrays
pandora_better = np.array(pandora_better)
closest_better = np.array(closest_better)
ties = np.array(ties)
same_clus = np.array(same_clus)


plt.figure(figsize=(8, 6))

# Pandora-associated cluster has better energy match
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[pandora_better],
    np.array(associated_cluster_delta_r_values)[pandora_better],
    s=10,
    alpha=0.5,
    label="Pandora cluster: better $E/p$ match"
)

# Nearest cluster has better energy match
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[closest_better],
    np.array(associated_cluster_delta_r_values)[closest_better],
    s=10,
    alpha=0.5,
    label="Nearest cluster: better $E/p$ match"
)

# Optional: exact ties
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[ties],
    np.array(associated_cluster_delta_r_values)[ties],
    s=10,
    alpha=0.5,
    label="Tie"
)


# y = x reference line
max_delta_r = max(
    max(nearest_cluster_delta_r_values),
    max(associated_cluster_delta_r_values)
)

plt.plot(
    [0, max_delta_r],
    [0, max_delta_r],
    linestyle="--",
    label=r"$y=x$"
)

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title(r"Which Cluster Has the Better $E/p$ Match?")

plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 0.45)
plt.ylim(0, 0.45)
plt.savefig(
    "associated_vs_nearest_cluster_energy_match_211_85.png",
    dpi=150
)

plt.close()

print("Pandora better:", np.sum(pandora_better))
print("Closest better:", np.sum(closest_better))
print("Ties:", np.sum(ties))
print("Total:", len(ties))
print("Same cluster among ties:", np.sum(same_clus))
print("Non-nearest cases:", np.sum(~np.array(associated_cluster_is_nearest)))  # should be 39
print("Ties where NOT nearest:", np.sum(ties & ~np.array(associated_cluster_is_nearest)))  # should reveal the extra 1

print(len(cluster_energy_over_track_p))
print(len(closest_cluster_energy_over_track_p))
print(len(associated_cluster_is_nearest))

for k in range(len(associated_cluster_is_nearest)):
    if associated_cluster_is_nearest[k] and not ties[k]:
        print("FOUND THE MISMATCH:")
        print(f"  event={debug_event_idx[k]}, pfo={debug_pfo_idx[k]}")
        print(f"  associated_cluster_index={debug_assoc_idx[k]}, nearest_cluster_index={debug_nearest_idx[k]}")
        print(f"  pandora E/p = {cluster_energy_over_track_p[k]}")
        print(f"  closest E/p = {closest_cluster_energy_over_track_p[k]}")
        print(f"  difference = {cluster_energy_over_track_p[k] - closest_cluster_energy_over_track_p[k]}")

print("\n-----------------------------------------")
print("Cluster Association Validation")
print("-----------------------------------------")
print(f"Tracks with a cluster: {num_tracks_with_cluster}")
print(f"Pandora cluster is nearest: {num_associated_is_nearest}")
print(f"Association fraction: {cluster_association_fraction:.4f}")
print(f"Association fraction (%): {100 * cluster_association_fraction:.2f}%")
print(f"Charged PFOs with a track but NO cluster: {pfos_with_track_no_cluster}")
print("-----------------------------------------")


# ---------------------------------------------------------
# Plot: SiTracks vs PFO multiplicity
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist2d(
    pfos_per_event,
    sitracks_per_event,
    bins=[50, 50],
    norm=LogNorm()
)

plt.xlabel("PFO multiplicity")
plt.ylabel("SiTrack multiplicity")
plt.title("SiTracks vs PFO Multiplicity")
plt.colorbar(label="Number of events")

plt.savefig(
    "sitracks_vs_pfo_multiplicity_211_85.png",
    dpi=150
)

plt.close()