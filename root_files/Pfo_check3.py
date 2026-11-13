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

# Cluster association validation (nearest cluster to track)
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
# ---------------------------------------------------------
# NEW: Track association validation (nearest track to cluster)
# ---------------------------------------------------------
associated_track_delta_r_values = []
nearest_track_delta_r_values = []
associated_track_is_nearest = []
cluster_energy_over_associated_track_p = []
cluster_energy_over_nearest_track_p = []
debug_event_idx2 = []
debug_pfo_idx2 = []
debug_assoc_track_idx = []
debug_nearest_track_idx = []
pfos_with_track_no_calo_state = 0

# ---------------------------------------------------------
# NEW: Standalone clone-track detection (independent of PFO
# association). For every pair of tracks in an event, flag
# whether they look like the same trajectory reconstructed
# twice, based on first-state (near-IP) parameters.
# ---------------------------------------------------------
clone_pairs_per_event = []
events_with_clone_pair = []
num_tracks_involved_in_clones_per_event = []
clone_pair_track_multiplicities = []  # multiplicity of the event, once per clone pair found

pfos_with_track_no_cluster = 0

def get_calo_face_state(event_i, track_index, location_target=4):
    sb = pandora_track_statesbegin[event_i][track_index]
    se = pandora_track_statesend[event_i][track_index]
    for s in range(sb, se):
        if track_states_location[event_i][s] == location_target:
            return s
    return None  # no calo-face state for this track — worth flagging, not silently


def get_track_spatial_info(event_i, track_index):
    """
    Mirror of the cluster-side geometry, but for a track.
    Returns (phi_spatial, theta_spatial, p, pT) at the track's calo-face
    state, or None if there's no calo-face state or omega is degenerate.
    """
    calo_state = get_calo_face_state(event_i, track_index)
    if calo_state is None:
        return None

    x = track_states_x[event_i][calo_state]
    y = track_states_y[event_i][calo_state]
    z = track_states_z[event_i][calo_state]

    phi_spatial = np.arctan2(y, x)
    theta_spatial = np.arctan2(np.sqrt(x**2 + y**2), z)

    state_begin = pandora_track_statesbegin[event_i][track_index]
    omega = track_states_omega[event_i][state_begin]
    if omega == 0:
        return None

    pT = abs(0.3 * B / omega / 1000)
    sin_theta = np.sin(theta_spatial)
    if sin_theta == 0:
        return None
    p = pT / sin_theta

    return phi_spatial, theta_spatial, p, pT


def get_track_first_state_params(event_i, track_index):
    """
    Track parameters at the FIRST state (near the IP), not the calo face.
    Two genuine clone/duplicate tracks should agree here too; two distinct
    particles that merely converge near the calorimeter should not.
    Returns a dict of phi, omega, tanLambda, and the reference point.
    """
    state_begin = pandora_track_statesbegin[event_i][track_index]
    return {
        "phi": track_states_phi[event_i][state_begin],
        "omega": track_states_omega[event_i][state_begin],
        "tanLambda": track_states_tanLambda[event_i][state_begin],
        "x": track_states_x[event_i][state_begin],
        "y": track_states_y[event_i][state_begin],
        "z": track_states_z[event_i][state_begin],
    }


def is_clone_pair(event_i, idx_a, idx_b,
                   phi_tol=1e-3, tanlambda_tol=1e-3, omega_rel_tol=1e-3):
    """
    Flag two tracks as likely clones of the same trajectory if their
    first-state (near-IP) phi, tanLambda, and omega all agree within
    tight tolerances. omega uses a relative tolerance since its scale
    depends on curvature/momentum.
    """
    sb_a = pandora_track_statesbegin[event_i][idx_a]
    se_a = pandora_track_statesend[event_i][idx_a]
    sb_b = pandora_track_statesbegin[event_i][idx_b]
    se_b = pandora_track_statesend[event_i][idx_b]

    if sb_a == se_a or sb_b == se_b:
        # one of the tracks has no states at all
        return False

    a = get_track_first_state_params(event_i, idx_a)
    b = get_track_first_state_params(event_i, idx_b)

    phi_close = abs(a["phi"] - b["phi"]) < phi_tol
    tanl_close = abs(a["tanLambda"] - b["tanLambda"]) < tanlambda_tol

    omega_denom = max(abs(a["omega"]), abs(b["omega"]), 1e-12)
    omega_close = abs(a["omega"] - b["omega"]) < omega_rel_tol * omega_denom

    return phi_close and tanl_close and omega_close


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
    print("========== ALL EVENTS KEYS ==========")

    for key in events.keys():
        print(key)

    print("=====================================")
    clusters = events["PandoraClusters"]
    mcparticles = events["MCParticles"]
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
    sitrack_indices = events["SiTracks_objIdx/SiTracks_objIdx.index"].array()
    #sitrack_chi2 = events["SiTracks_Refitted/SiTracks_Refitted.chi2"].array()
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

        # ---------------------------------------------------------
        # NEW: Clone-track detection, independent of PFO association.
        # Check every track pair in the event; flag likely clones.
        # ---------------------------------------------------------
        clone_pair_count = 0
        tracks_in_clone_pair = set()

        for a_idx in range(num_total_tracks):
            for b_idx in range(a_idx + 1, num_total_tracks):
                if is_clone_pair(i, a_idx, b_idx):
                    clone_pair_count += 1
                    tracks_in_clone_pair.add(a_idx)
                    tracks_in_clone_pair.add(b_idx)
                    clone_pair_track_multiplicities.append(num_total_tracks)

        clone_pairs_per_event.append(clone_pair_count)
        events_with_clone_pair.append(clone_pair_count > 0)
        num_tracks_involved_in_clones_per_event.append(len(tracks_in_clone_pair))

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

            # ===========================================================
            # NEW: Track association validation (reverse direction)
            #
            # Now fix the ASSOCIATED CLUSTER's position, and ask: among
            # ALL tracks in this event, which one is geometrically
            # closest to that cluster? Is it the track Pandora actually
            # linked to this PFO?
            # ===========================================================

            nearest_track_delta_r = np.inf
            nearest_track_index = None
            nearest_track_p = None

            num_tracks_this_event = len(pandora_track_statesbegin[i])

            for track_index_all in range(num_tracks_this_event):

                track_info = get_track_spatial_info(i, track_index_all)
                if track_info is None:
                    # no calo-face state (or degenerate omega) for this track
                    continue

                phi_all, theta_all, p_all, pT_all = track_info

                delta_phi_trk = np.arctan2(
                    np.sin(associated_cluster_phi - phi_all),
                    np.cos(associated_cluster_phi - phi_all)
                )
                delta_theta_trk = associated_cluster_theta - theta_all
                delta_r_trk = np.sqrt(delta_phi_trk**2 + delta_theta_trk**2)

                if delta_r_trk < nearest_track_delta_r:
                    nearest_track_delta_r = delta_r_trk
                    nearest_track_index = track_index_all
                    nearest_track_p = p_all

            if nearest_track_index is None:
                # No track in this event had a usable calo-face state.
                pfos_with_track_no_calo_state += 1
            else:
                # Note: this deltaR is the same track-cluster pair as
                # associated_delta_r above (deltaR is symmetric), kept
                # separately so this block reads standalone.
                associated_track_delta_r_values.append(associated_delta_r)
                nearest_track_delta_r_values.append(nearest_track_delta_r)

                is_nearest_track = (track_index == nearest_track_index)
                associated_track_is_nearest.append(is_nearest_track)

                cluster_energy_over_associated_track_p.append(
                    pfo_cluster_energy / p_track
                )
                cluster_energy_over_nearest_track_p.append(
                    pfo_cluster_energy / nearest_track_p
                    if nearest_track_p not in (None, 0) else np.nan
                )

                debug_event_idx2.append(i)
                debug_pfo_idx2.append(pfo)
                debug_assoc_track_idx.append(track_index)
                debug_nearest_track_idx.append(nearest_track_index)

            # ---------------------------------------------------------
            # PFO quantities
            # ---------------------------------------------------------
            px_pfo = pfo_momx[i][pfo]
            py_pfo = pfo_momy[i][pfo]

            pT_pfo = np.sqrt(px_pfo**2 + py_pfo**2)
            E_pfo = pfo_energy[i][pfo]

            # ---------------------------------------------------------
            # Validation quantities
            # ---------------------------------------------------------
            delta_phi = associated_delta_phi
            delta_theta = associated_delta_theta

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

            # Don't need to look at another AllTracks relation
            # if the PFO should have only one track.

        tracks_per_event.append(num_total_tracks)
        pfos_per_event.append(num_total_pfos)
        num_si_tracks = len(sitrack_indices[i])
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
plt.hist(pfo_energy_over_track_pt, bins=50, histtype="step")
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
plt.hist(pfo_pt_over_track_pt, bins=50, histtype="step")
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
plt.hist(cluster_energy_over_track_pt, bins=50, histtype="step")
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
plt.hist(delta_phi_values, bins=50, histtype="step")
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
plt.hist(delta_theta_values, bins=50, histtype="step")
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
plt.hist2d(delta_phi_values, delta_theta_values, bins=50, norm=LogNorm())
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
plt.hist(pfo_energy_over_track_p, bins=50, histtype="step")
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
plt.hist(cluster_energy_over_track_p, bins=50, histtype="step")
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
plt.hist2d(pfos_per_event, associated_tracks_per_event, bins=[50, 50], norm=LogNorm())
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
plt.hist2d(tracks_per_event, associated_tracks_per_event, bins=[50, 50], norm=LogNorm())
plt.xlabel("Track multiplicity")
plt.ylabel("Number of tracks associated with PFOs")
plt.title("Associated Tracks vs Track Multiplicity")
plt.colorbar(label="Number of events")
plt.savefig("associated_tracks_vs_track_multiplicity_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot: Track-PFO association fraction per event
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(associated_fraction_per_event, bins=50, histtype="step")
plt.xlabel(r"$N_{\mathrm{tracks\ with\ a\ PFO}} / N_{\mathrm{tracks}}$")
plt.ylabel("Number of events")
plt.title("Track-to-PFO Association Efficiency")
plt.xlim(0, 1)
plt.grid(alpha=0.3)
plt.savefig("track_pfo_association_fraction_per_event_211_85.png", dpi=150)
plt.close()

#Cluster Energy vs Track pT
plt.figure(figsize=(8, 6))
plt.hist2d(track_pt_values, cluster_energy_values, bins=[50, 50], norm=LogNorm())
plt.xlabel(r"$p_T^{\mathrm{track}}$ [GeV]")
plt.ylabel(r"$E_{\mathrm{cluster}}$ [GeV]")
plt.title("Cluster Energy vs Track $p_T$")
plt.colorbar(label="Number of charged PFOs")
plt.savefig("cluster_energy_vs_track_pt_211_85.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
plt.hist2d(track_p_values, cluster_energy_values, bins=[50, 50], norm=LogNorm())
plt.xlabel(r"$p^{\mathrm{track}}$ [GeV]")
plt.ylabel(r"$E_{\mathrm{cluster}}$ [GeV]")
plt.title("Cluster Energy vs Track $p$")
plt.colorbar(label="Number of charged PFOs")
plt.savefig("cluster_energy_vs_track_p_211_85.png", dpi=150)
plt.close()

#Lastlly we will do a nearest cluster delta R versus delta R of the cluster Pandora actually associates with the PFO
#We will do N(pandora associated cluster is nearest to track) / N (tracks with a cluster)

# ---------------------------------------------------------
# Plot 11: Pandora-associated cluster deltaR vs
# nearest cluster deltaR
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(nearest_cluster_delta_r_values, associated_cluster_delta_r_values, bins=50, norm=LogNorm())

max_delta_r = max(max(nearest_cluster_delta_r_values), max(associated_cluster_delta_r_values))
plt.plot([0, max_delta_r], [0, max_delta_r], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title("Pandora Cluster Association vs Nearest Cluster")
plt.colorbar(label="Number of charged PFOs")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("associated_vs_nearest_cluster_deltaR_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Cluster association fraction
# ---------------------------------------------------------

num_tracks_with_cluster = len(associated_cluster_is_nearest)
num_associated_is_nearest = np.sum(associated_cluster_is_nearest)

if num_tracks_with_cluster > 0:

    cluster_association_fraction = (
        num_associated_is_nearest / num_tracks_with_cluster
    )

    print("\n-----------------------------------------")
    print("Cluster Association Validation")
    print("-----------------------------------------")
    print(f"Tracks with a cluster: {num_tracks_with_cluster}")
    print(f"Pandora cluster is nearest: {num_associated_is_nearest}")
    print(f"Association fraction: {cluster_association_fraction:.4f}")
    print(f"Association fraction (%): {100 * cluster_association_fraction:.2f}%")
    print("-----------------------------------------")

# ---------------------------------------------------------
# Plot 12: Pandora-associated cluster vs nearest cluster
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(associated_cluster_is_nearest, bins=[-0.5, 0.5, 1.5], histtype="step")
plt.xticks([0, 1], ["Not nearest", "Nearest"])
plt.xlabel("Is Pandora-associated cluster the nearest cluster?")
plt.ylabel("Number of charged PFOs")
plt.title("Track--Cluster Association")
plt.grid(alpha=0.3)
plt.savefig("pandora_cluster_association_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 13: Pandora-associated cluster deltaR vs
# nearest cluster deltaR
#
# Color = E_cluster / p_track
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    nearest_cluster_delta_r_values,
    associated_cluster_delta_r_values,
    c=associated_cluster_energy_over_track_p,
    s=10, alpha=0.5, norm=LogNorm()
)

max_delta_r = max(max(nearest_cluster_delta_r_values), max(associated_cluster_delta_r_values))
plt.plot([0, max_delta_r], [0, max_delta_r], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title(r"Cluster Association: $\Delta R$ Comparison")
plt.colorbar(scatter, label=r"$E_{\mathrm{cluster}} / p_{\mathrm{track}}$")
plt.legend()
plt.grid(alpha=0.3)
#plt.xlim(0, 0.45)
#plt.ylim(0, 0.45)
plt.savefig("associated_vs_nearest_cluster_deltaR_colored_energy_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 14: Pandora-cluster deltaR vs
# nearest cluster deltaR
#
# Color = E_closest_cluster / p_track
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    nearest_cluster_delta_r_values,
    associated_cluster_delta_r_values,
    c=closest_cluster_energy_over_track_p,
    s=10, alpha=0.5, norm=LogNorm()
)

max_delta_r = max(max(nearest_cluster_delta_r_values), max(associated_cluster_delta_r_values))
plt.plot([0, max_delta_r], [0, max_delta_r], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title(r"Cluster Association: $\Delta R$ Comparison")
plt.colorbar(scatter, label=r"$E_{\mathrm{cluster}} (closest cluster) / p_{\mathrm{track}}$")
plt.legend()
plt.grid(alpha=0.3)
#plt.xlim(0, 0.45)
#plt.ylim(0, 0.45)
plt.savefig("associated_vs_nearest_cluster_deltaR_colored_closest_energy_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 15: Which cluster has the better energy match?
#
# Better match = whichever cluster's E/p is closer to 1
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
        same_clus.append(is_nearest)

pandora_better = np.array(pandora_better)
closest_better = np.array(closest_better)
ties = np.array(ties)
same_clus = np.array(same_clus)

plt.figure(figsize=(8, 6))
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[pandora_better],
    np.array(associated_cluster_delta_r_values)[pandora_better],
    s=10, alpha=0.5, label="Pandora cluster: better $E/p$ match"
)
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[closest_better],
    np.array(associated_cluster_delta_r_values)[closest_better],
    s=10, alpha=0.5, label="Nearest cluster: better $E/p$ match"
)
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[ties],
    np.array(associated_cluster_delta_r_values)[ties],
    s=10, alpha=0.5, label="Tie"
)

max_delta_r = max(max(nearest_cluster_delta_r_values), max(associated_cluster_delta_r_values))
plt.plot([0, max_delta_r], [0, max_delta_r], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ cluster}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ cluster}}$")
plt.title(r"Which Cluster Has the Better $E/p$ Match?")
plt.legend()
plt.grid(alpha=0.3)
#plt.xlim(0, 0.45)
#plt.ylim(0, 0.45)
plt.savefig("associated_vs_nearest_cluster_energy_match_211_85.png", dpi=150)
plt.close()

print("Pandora better:", np.sum(pandora_better))
print("Closest better:", np.sum(closest_better))
print("Ties:", np.sum(ties))
print("Total:", len(ties))
print("Same cluster among ties:", np.sum(same_clus))
print("Non-nearest cases:", np.sum(~np.array(associated_cluster_is_nearest)))
print("Ties where NOT nearest:", np.sum(ties & ~np.array(associated_cluster_is_nearest)))

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

# ===========================================================
# NEW SECTION: Track association validation (reverse direction)
#
# For each PFO's associated cluster, is the track Pandora linked
# to it actually the nearest track in the event to that cluster?
# ===========================================================

# ---------------------------------------------------------
# Plot 11b: Pandora-associated track deltaR vs
# nearest track deltaR
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(nearest_track_delta_r_values, associated_track_delta_r_values, bins=50, norm=LogNorm())

max_delta_r_trk = max(max(nearest_track_delta_r_values), max(associated_track_delta_r_values))
plt.plot([0, max_delta_r_trk], [0, max_delta_r_trk], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ track}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ track}}$")
plt.title("Pandora Track Association vs Nearest Track")
plt.colorbar(label="Number of charged PFOs")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("tr_associated_vs_nearest_track_deltaR_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Track association fraction
# ---------------------------------------------------------

num_clusters_with_track = len(associated_track_is_nearest)
num_track_is_nearest = np.sum(associated_track_is_nearest)

if num_clusters_with_track > 0:

    track_association_fraction = (
        num_track_is_nearest / num_clusters_with_track
    )

    print("\n-----------------------------------------")
    print("Track Association Validation (reverse direction)")
    print("-----------------------------------------")
    print(f"Clusters with an associated track: {num_clusters_with_track}")
    print(f"Pandora track is nearest: {num_track_is_nearest}")
    print(f"Association fraction: {track_association_fraction:.4f}")
    print(f"Association fraction (%): {100 * track_association_fraction:.2f}%")
    print(f"Cases with no usable track in event: {pfos_with_track_no_calo_state}")
    print("-----------------------------------------")

# ---------------------------------------------------------
# Plot 12b: Pandora-associated track vs nearest track
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(associated_track_is_nearest, bins=[-0.5, 0.5, 1.5], histtype="step")
plt.xticks([0, 1], ["Not nearest", "Nearest"])
plt.xlabel("Is Pandora-associated track the nearest track?")
plt.ylabel("Number of charged PFOs")
plt.title("Cluster--Track Association")
plt.grid(alpha=0.3)
plt.savefig("tr_pandora_track_association_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 13b: Pandora-associated track deltaR vs
# nearest track deltaR
#
# Color = E_cluster / p_(associated track)
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    nearest_track_delta_r_values,
    associated_track_delta_r_values,
    c=cluster_energy_over_associated_track_p,
    s=10, alpha=0.5, norm=LogNorm()
)

plt.plot([0, max_delta_r_trk], [0, max_delta_r_trk], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ track}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ track}}$")
plt.title(r"Track Association: $\Delta R$ Comparison")
plt.colorbar(scatter, label=r"$E_{\mathrm{cluster}} / p_{\mathrm{associated\ track}}$")
plt.legend()
plt.grid(alpha=0.3)
#plt.xlim(0, 0.45)
#plt.ylim(0, 0.45)
plt.savefig("tr_associated_vs_nearest_track_deltaR_colored_energy_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 14b: Pandora-associated track deltaR vs
# nearest track deltaR
#
# Color = E_cluster / p_(nearest track)
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    nearest_track_delta_r_values,
    associated_track_delta_r_values,
    c=cluster_energy_over_nearest_track_p,
    s=10, alpha=0.5, norm=LogNorm()
)

plt.plot([0, max_delta_r_trk], [0, max_delta_r_trk], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ track}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ track}}$")
plt.title(r"Track Association: $\Delta R$ Comparison")
plt.colorbar(scatter, label=r"$E_{\mathrm{cluster}} / p_{\mathrm{nearest\ track}}$")
plt.legend()
plt.grid(alpha=0.3)
#plt.xlim(0, 0.45)
#plt.ylim(0, 0.45)
plt.savefig("tr_associated_vs_nearest_track_deltaR_colored_nearest_energy_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 15b: Which track gives the better E/p match?
#
# Better match = whichever track's E_cluster/p is closer to 1
# ---------------------------------------------------------

pandora_track_better = []
nearest_track_better = []
track_ties = []
track_same = []

for pandora_ep, nearest_ep, is_nearest in zip(
    cluster_energy_over_associated_track_p,
    cluster_energy_over_nearest_track_p,
    associated_track_is_nearest
):
    if np.isnan(nearest_ep):
        pandora_track_better.append(False)
        nearest_track_better.append(False)
        track_ties.append(False)
        continue

    pandora_distance = abs(pandora_ep - 1.0)
    nearest_distance = abs(nearest_ep - 1.0)

    if pandora_distance < nearest_distance:
        pandora_track_better.append(True)
        nearest_track_better.append(False)
        track_ties.append(False)
    elif nearest_distance < pandora_distance:
        pandora_track_better.append(False)
        nearest_track_better.append(True)
        track_ties.append(False)
    else:
        pandora_track_better.append(False)
        nearest_track_better.append(False)
        track_ties.append(True)
        track_same.append(is_nearest)

pandora_track_better = np.array(pandora_track_better)
nearest_track_better = np.array(nearest_track_better)
track_ties = np.array(track_ties)
track_same = np.array(track_same)

plt.figure(figsize=(8, 6))
plt.scatter(
    np.array(nearest_track_delta_r_values)[pandora_track_better],
    np.array(associated_track_delta_r_values)[pandora_track_better],
    s=10, alpha=0.5, label="Pandora track: better $E/p$ match"
)
plt.scatter(
    np.array(nearest_track_delta_r_values)[nearest_track_better],
    np.array(associated_track_delta_r_values)[nearest_track_better],
    s=10, alpha=0.5, label="Nearest track: better $E/p$ match"
)
plt.scatter(
    np.array(nearest_track_delta_r_values)[track_ties],
    np.array(associated_track_delta_r_values)[track_ties],
    s=10, alpha=0.5, label="Tie"
)

plt.plot([0, max_delta_r_trk], [0, max_delta_r_trk], linestyle="--", label=r"$y=x$")

plt.xlabel(r"$\Delta R_{\mathrm{nearest\ track}}$")
plt.ylabel(r"$\Delta R_{\mathrm{Pandora\ associated\ track}}$")
plt.title(r"Which Track Has the Better $E/p$ Match?")
plt.legend()
plt.grid(alpha=0.3)
#plt.xlim(0, 0.45)
#plt.ylim(0, 0.45)
plt.savefig("tr_associated_vs_nearest_track_energy_match_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 16b: ΔR(nearest track) - ΔR(associated track)
#
# This is >= 0 by construction (nearest is the minimum over
# all tracks). If mismatches are near-degenerate duplicate
# track candidates, this should pile up right at 0. A separate
# population at larger differences would flag genuine
# mis-associations worth chasing individually.
# ---------------------------------------------------------

track_delta_r_diff = (
    np.array(nearest_track_delta_r_values)
    - np.array(associated_track_delta_r_values)
)

plt.figure(figsize=(8, 6))
plt.hist(track_delta_r_diff, bins=100, histtype="step")
plt.xlabel(r"$\Delta R_{\mathrm{nearest\ track}} - \Delta R_{\mathrm{associated\ track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("Track Association: Nearest vs Associated $\\Delta R$ Gap")
plt.yscale("log")
plt.grid(alpha=0.3)
plt.savefig("tr_track_deltaR_gap_full_range_211_85.png", dpi=150)
plt.close()

# Zoomed-in version, since mismatches/ties should sit very close to 0
plt.figure(figsize=(8, 6))
plt.hist(track_delta_r_diff, bins=100, histtype="step", range=(0, 0.001))
plt.xlabel(r"$\Delta R_{\mathrm{nearest\ track}} - \Delta R_{\mathrm{associated\ track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("Track Association: Nearest vs Associated $\\Delta R$ Gap (zoomed)")
plt.grid(alpha=0.3)
plt.savefig("tr_track_deltaR_gap_zoomed_211_85.png", dpi=150)
plt.close()

print("\n-----------------------------------------")
print("Track deltaR Gap Summary")
print("-----------------------------------------")
print(f"Median gap: {np.median(track_delta_r_diff):.6f}")
print(f"Mean gap:   {np.mean(track_delta_r_diff):.6f}")
print(f"Fraction of cases with gap < 1e-4: "
      f"{np.mean(track_delta_r_diff < 1e-4):.4f}")
print(f"Fraction of cases with gap < 1e-5: "
      f"{np.mean(track_delta_r_diff < 1e-5):.4f}")
print("-----------------------------------------")

print("\n-----------------------------------------")
print("Track E/p Match Summary")
print("-----------------------------------------")
print("Pandora track better:", np.sum(pandora_track_better))
print("Nearest track better:", np.sum(nearest_track_better))
print("Ties:", np.sum(track_ties))
print("Total:", len(track_ties))
print("Same track among ties:", np.sum(track_same))
print("Non-nearest cases:", np.sum(~np.array(associated_track_is_nearest)))
print("Ties where NOT nearest:", np.sum(track_ties & ~np.array(associated_track_is_nearest)))

for k in range(len(associated_track_is_nearest)):
    if not associated_track_is_nearest[k] and not track_ties[k]:
        mismatch_event = debug_event_idx2[k]
        num_tracks_in_event = len(pandora_track_statesbegin[mismatch_event])
        delta_r_diff = (
            nearest_track_delta_r_values[k] - associated_track_delta_r_values[k]
        )

        print("FOUND A TRACK MISMATCH:")
        print(f"  event={mismatch_event}, pfo={debug_pfo_idx2[k]}")
        print(f"  associated_track_index={debug_assoc_track_idx[k]}, nearest_track_index={debug_nearest_track_idx[k]}")
        print(f"  ΔR (associated track) = {associated_track_delta_r_values[k]:.6f}")
        print(f"  ΔR (nearest track)    = {nearest_track_delta_r_values[k]:.6f}")
        print(f"  ΔR difference         = {delta_r_diff:.6f}")
        print(f"  Track multiplicity in event = {num_tracks_in_event}")
        print(f"  E/p (associated track) = {cluster_energy_over_associated_track_p[k]}")
        print(f"  E/p (nearest track)    = {cluster_energy_over_nearest_track_p[k]}")

        # ---------------------------------------------------------
        # First-state (near-IP) comparison.
        #
        # True clone/duplicate tracks should agree here too, not just
        # at the calo face. Distinct particles that happen to converge
        # near the calorimeter should look different at the IP.
        # ---------------------------------------------------------
        assoc_first = get_track_first_state_params(
            mismatch_event, debug_assoc_track_idx[k]
        )
        nearest_first = get_track_first_state_params(
            mismatch_event, debug_nearest_track_idx[k]
        )

        print("  --- First-state (IP) comparison ---")
        print(
            f"  phi:       assoc={assoc_first['phi']:.6f}, "
            f"nearest={nearest_first['phi']:.6f}, "
            f"diff={assoc_first['phi'] - nearest_first['phi']:.6f}"
        )
        print(
            f"  omega:     assoc={assoc_first['omega']:.6e}, "
            f"nearest={nearest_first['omega']:.6e}, "
            f"diff={assoc_first['omega'] - nearest_first['omega']:.6e}"
        )
        print(
            f"  tanLambda: assoc={assoc_first['tanLambda']:.6f}, "
            f"nearest={nearest_first['tanLambda']:.6f}, "
            f"diff={assoc_first['tanLambda'] - nearest_first['tanLambda']:.6f}"
        )
        print(
            f"  ref point: assoc=({assoc_first['x']:.3f}, {assoc_first['y']:.3f}, {assoc_first['z']:.3f}), "
            f"nearest=({nearest_first['x']:.3f}, {nearest_first['y']:.3f}, {nearest_first['z']:.3f})"
        )

        first_state_looks_like_clone = (
            abs(assoc_first["phi"] - nearest_first["phi"]) < 1e-3
            and abs(assoc_first["tanLambda"] - nearest_first["tanLambda"]) < 1e-3
            and abs(assoc_first["omega"] - nearest_first["omega"]) < 1e-3 * abs(assoc_first["omega"])
        )
        print(f"  Looks like a clone at the IP too: {first_state_looks_like_clone}")

print("\n-----------------------------------------")
print("Track Association Validation (reverse direction)")
print("-----------------------------------------")
print(f"Clusters with an associated track: {num_clusters_with_track}")
print(f"Pandora track is nearest: {num_track_is_nearest}")
print(f"Association fraction: {track_association_fraction:.4f}")
print(f"Association fraction (%): {100 * track_association_fraction:.2f}%")
print(f"Cases with no usable track in event: {pfos_with_track_no_calo_state}")
print("-----------------------------------------")

# ===========================================================
# NEW SECTION: Clone-track detection (standalone, independent
# of PFO/cluster association)
# ===========================================================

clone_pairs_per_event = np.array(clone_pairs_per_event)
events_with_clone_pair = np.array(events_with_clone_pair)
num_tracks_involved_in_clones_per_event = np.array(num_tracks_involved_in_clones_per_event)
tracks_per_event_arr = np.array(tracks_per_event)

# ---------------------------------------------------------
# Plot: clone pairs per event
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
max_clone_pairs = max(1, int(clone_pairs_per_event.max()))
plt.hist(
    clone_pairs_per_event,
    bins=np.arange(-0.5, max_clone_pairs + 1.5, 1),
    histtype="step"
)
plt.xlabel("Number of likely clone track pairs")
plt.ylabel("Number of events")
plt.title("Clone Track Pairs per Event")
plt.yscale("log")
plt.grid(alpha=0.3)
plt.savefig("tr_clone_track_pairs_per_event_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot: track multiplicity vs number of clone pairs
# (are clones concentrated at high multiplicity, or spread out?)
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(
    tracks_per_event_arr,
    clone_pairs_per_event,
    bins=[
        np.arange(-0.5, tracks_per_event_arr.max() + 1.5, 1),
        np.arange(-0.5, max_clone_pairs + 1.5, 1),
    ],
    norm=LogNorm()
)
plt.xlabel("Track multiplicity")
plt.ylabel("Number of likely clone track pairs")
plt.title("Clone Pairs vs Track Multiplicity")
plt.colorbar(label="Number of events")
plt.savefig("tr_clone_pairs_vs_track_multiplicity_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot: fraction of tracks in an event that are part of
# a clone pair
# ---------------------------------------------------------

with np.errstate(divide="ignore", invalid="ignore"):
    clone_track_fraction_per_event = np.where(
        tracks_per_event_arr > 0,
        num_tracks_involved_in_clones_per_event / tracks_per_event_arr,
        0.0
    )

plt.figure(figsize=(8, 6))
plt.hist(clone_track_fraction_per_event, bins=50, histtype="step")
plt.xlabel(r"$N_{\mathrm{tracks\ in\ a\ clone\ pair}} / N_{\mathrm{tracks}}$")
plt.ylabel("Number of events")
plt.title("Fraction of Tracks Involved in a Clone Pair")
plt.xlim(0, 1)
plt.grid(alpha=0.3)
plt.savefig("tr_clone_track_fraction_per_event_211_85.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

num_events_total = len(clone_pairs_per_event)
num_events_with_clone = np.sum(events_with_clone_pair)
num_events_multi_track = np.sum(tracks_per_event_arr >= 2)

print("\n-----------------------------------------")
print("Clone Track Detection Summary")
print("-----------------------------------------")
print(f"Total events: {num_events_total}")
print(f"Events with >=2 tracks (eligible for a clone pair): {num_events_multi_track}")
print(f"Events with at least one likely clone pair: {num_events_with_clone}")
if num_events_multi_track > 0:
    print(
        f"Clone rate among multi-track events: "
        f"{100 * num_events_with_clone / num_events_multi_track:.2f}%"
    )
print(f"Total likely clone pairs found: {np.sum(clone_pairs_per_event)}")
print(f"Mean clone pairs per event: {np.mean(clone_pairs_per_event):.4f}")
if len(clone_pair_track_multiplicities) > 0:
    print(
        f"Mean track multiplicity in events containing a clone pair: "
        f"{np.mean(clone_pair_track_multiplicities):.2f}"
    )
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


print(
    f"Event {i}: "
    f"PFOs = {num_total_pfos}, "
    f"associated AllTracks = {len(associated_track_indices)}, "
    f"SiTracks = {len(sitrack_indices[i])}"
)

# ---------------------------------------------------------
# Check for mismatch between associated tracks and SiTracks
# ---------------------------------------------------------

for i, (n_associated, n_si) in enumerate(
    zip(associated_tracks_per_event, sitracks_per_event)
):
    if n_associated != n_si:
        print(
            f"PROBLEM: Event {i}: "
            f"associated AllTracks = {n_associated}, "
            f"SiTracks = {n_si}"
        )