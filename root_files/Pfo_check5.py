#Debug script for a single event (946): dump cluster and track positions/momenta,
#print per-PFO matched-cluster E/p, and plot cluster phi/theta positions labeled in red.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

EVENT_INDEX = 961

file = uproot.open(
    "/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_65-65_testing_fede2/"
    "job_0/reco_output_p50_211_nobib0.edm4hep.root"
)
events = file["events"]

clusters = events["PandoraClusters"]
cluster_energies = clusters["PandoraClusters.energy"].array()
cluster_x = events["PandoraClusters/PandoraClusters.position.x"].array()
cluster_y = events["PandoraClusters/PandoraClusters.position.y"].array()
cluster_z = events["PandoraClusters/PandoraClusters.position.z"].array()

pfo_energy = events["PandoraPFOs/PandoraPFOs.energy"].array()
pfo_momx = events["PandoraPFOs/PandoraPFOs.momentum.x"].array()
pfo_momy = events["PandoraPFOs/PandoraPFOs.momentum.y"].array()
pfo_momz = events["PandoraPFOs/PandoraPFOs.momentum.z"].array()
pandora_tracks_begin = events["PandoraPFOs/PandoraPFOs.tracks_begin"].array()
pandora_tracks_end = events["PandoraPFOs/PandoraPFOs.tracks_end"].array()
pandora_clusters_begin = events["PandoraPFOs/PandoraPFOs.clusters_begin"].array()
pandora_clusters_end = events["PandoraPFOs/PandoraPFOs.clusters_end"].array()
pandora_clusters_index = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"].array()
pandora_tracks_index = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"].array()
pandora_tracks_collection = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.collectionID"].array()

pandora_track_statesbegin = events["AllTracks/AllTracks.trackStates_begin"].array()
pandora_track_statesend = events["AllTracks/AllTracks.trackStates_end"].array()
track_states_omega = events["_AllTracks_trackStates/_AllTracks_trackStates.omega"].array()
track_states_x = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.x"].array()
track_states_y = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.y"].array()
track_states_z = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.z"].array()
track_states_location = events["_AllTracks_trackStates/_AllTracks_trackStates.location"].array()

sitrack_indices = events["SiTracks_objIdx/SiTracks_objIdx.index"].array()

B = 5


def get_calo_face_state(event_i, track_index, location_target=4):
    sb = pandora_track_statesbegin[event_i][track_index]
    se = pandora_track_statesend[event_i][track_index]
    for s in range(sb, se):
        if track_states_location[event_i][s] == location_target:
            return s
    return None


def get_track_phi_theta_p(event_i, track_index):
    """Returns (phi, theta, p) at the calo-face state, or None if unavailable."""
    calo_state = get_calo_face_state(event_i, track_index)
    if calo_state is None:
        return None

    x = track_states_x[event_i][calo_state]
    y = track_states_y[event_i][calo_state]
    z = track_states_z[event_i][calo_state]

    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)

    state_begin = pandora_track_statesbegin[event_i][track_index]
    omega = track_states_omega[event_i][state_begin]
    if omega == 0:
        return None

    pT = abs(0.3 * B / omega / 1000)
    sin_theta = np.sin(theta)
    if sin_theta == 0:
        return None
    p = pT / sin_theta

    return phi, theta, p


i = EVENT_INDEX

print(f"\n===== EVENT {i} =====\n")

# ---------------------------------------------------------
# 1. All clusters in this event: phi, theta, energy
# ---------------------------------------------------------
print("----- Clusters -----")
n_clusters_event = len(cluster_energies[i])
cluster_phis = []
cluster_thetas = []
for c in range(n_clusters_event):
    x = cluster_x[i][c]
    y = cluster_y[i][c]
    z = cluster_z[i][c]
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    energy = cluster_energies[i][c]
    cluster_phis.append(phi)
    cluster_thetas.append(theta)
    print(f"Cluster {c}: phi={phi:.4f} rad, theta={theta:.4f} rad, E={energy:.4f} GeV")

# ---------------------------------------------------------
# 2. Per-PFO: matched track momentum, matched cluster(s) E/p
# ---------------------------------------------------------
print("\n----- PFOs: matched track + matched cluster E/p -----")

associated_track_indices_946 = set()

for pfo in range(len(pfo_energy[i])):
    track_begin = pandora_tracks_begin[i][pfo]
    track_end = pandora_tracks_end[i][pfo]

    alltracks_per_pfo = []
    for j in range(track_begin, track_end):
        track_index = pandora_tracks_index[i][j]
        collection_id = pandora_tracks_collection[i][j]
        if collection_id == 4111979324:
            alltracks_per_pfo.append(track_index)

    if len(alltracks_per_pfo) != 1:
        print(f"PFO {pfo}: skipping (found {len(alltracks_per_pfo)} AllTracks tracks, expected 1)")
        continue

    track_index = alltracks_per_pfo[0]
    associated_track_indices_946.add(track_index)

    track_info = get_track_phi_theta_p(i, track_index)
    if track_info is None:
        print(f"PFO {pfo}: track {track_index} has no usable calo-face state — skipping")
        continue
    phi_track, theta_track, p_track = track_info

    cluster_begin = pandora_clusters_begin[i][pfo]
    cluster_end = pandora_clusters_end[i][pfo]
    if cluster_begin == cluster_end:
        print(f"PFO {pfo}: track {track_index} (p={p_track:.4f} GeV) has no cluster")
        continue

    pfo_cluster_energy = 0.0
    matched_cluster_indices = []
    for j in range(cluster_begin, cluster_end):
        cluster_index = pandora_clusters_index[i][j]
        matched_cluster_indices.append(cluster_index)
        pfo_cluster_energy += cluster_energies[i][cluster_index]

    e_over_p = pfo_cluster_energy / p_track if p_track != 0 else np.nan

    print(
        f"PFO {pfo}: track {track_index} "
        f"(p={p_track:.4f} GeV, phi={phi_track:.4f}, theta={theta_track:.4f}) "
        f"-> cluster(s) {matched_cluster_indices}, "
        f"E_cluster={pfo_cluster_energy:.4f} GeV, E/p={e_over_p:.4f}"
    )

# ---------------------------------------------------------
# 3. Tracks referenced in "associated AllTracks" for this event
# ---------------------------------------------------------
print("\n----- Associated AllTracks (matched to some PFO) -----")
for track_index in sorted(associated_track_indices_946):
    track_info = get_track_phi_theta_p(i, track_index)
    if track_info is None:
        print(f"AllTracks index {track_index}: no usable calo-face state")
        continue
    phi, theta, p = track_info
    print(f"AllTracks index {track_index}: p={p:.4f} GeV, phi={phi:.4f}, theta={theta:.4f}")

# ---------------------------------------------------------
# 4. All SiTracks for this event
# ---------------------------------------------------------
print("\n----- SiTracks -----")
sitrack_index_set_946 = set(sitrack_indices[i])
for track_index in sorted(sitrack_index_set_946):
    track_info = get_track_phi_theta_p(i, track_index)
    if track_info is None:
        print(f"SiTrack index {track_index}: no usable calo-face state")
        continue
    phi, theta, p = track_info
    print(f"SiTrack index {track_index}: p={p:.4f} GeV, phi={phi:.4f}, theta={theta:.4f}")

missing = sitrack_index_set_946 - associated_track_indices_946
if missing:
    print(f"\nSiTracks NOT claimed by any PFO in event {i}: {sorted(missing)}")
else:
    print(f"\nAll SiTracks in event {i} were claimed by a PFO.")

# ---------------------------------------------------------
# 4b. Compare every PFO's pT/phi/theta against every SiTrack's
# pT/phi/theta, regardless of whether Pandora linked them via an
# AllTracks relation. This is meant to catch cases where a PFO's
# kinematics closely match an "unclaimed" SiTrack -- suggesting the
# AllTracks relation itself is missing/broken rather than there being
# no real correspondence between the two.
# ---------------------------------------------------------
print("\n----- PFO pT vs SiTrack pT/phi/theta comparison -----")

pfo_kinematics = []  # (pfo_idx, pt_pfo, phi_pfo, theta_pfo)
for pfo in range(len(pfo_energy[i])):
    px = pfo_momx[i][pfo]
    py = pfo_momy[i][pfo]
    pz = pfo_momz[i][pfo]
    pt_pfo = np.sqrt(px**2 + py**2)
    phi_pfo = np.arctan2(py, px)
    theta_pfo = np.arctan2(pt_pfo, pz)
    pfo_kinematics.append((pfo, pt_pfo, phi_pfo, theta_pfo))

sitrack_kinematics = []  # (track_index, pt_si, phi_si, theta_si)
for track_index in sorted(sitrack_index_set_946):
    track_info = get_track_phi_theta_p(i, track_index)
    if track_info is None:
        continue
    phi_si, theta_si, p_si = track_info
    pt_si = p_si * np.sin(theta_si)
    sitrack_kinematics.append((track_index, pt_si, phi_si, theta_si))

if not sitrack_kinematics:
    print("No SiTracks with a usable calo-face state to compare against.")
else:
    for pfo, pt_pfo, phi_pfo, theta_pfo in pfo_kinematics:
        for track_index, pt_si, phi_si, theta_si in sitrack_kinematics:
            delta_phi = np.arctan2(
                np.sin(phi_pfo - phi_si),
                np.cos(phi_pfo - phi_si),
            )
            delta_theta = theta_pfo - theta_si
            pt_ratio = pt_pfo / pt_si if pt_si != 0 else np.nan

            print(
                f"PFO {pfo} (pt={pt_pfo:.4f} GeV, phi={phi_pfo:.4f}, theta={theta_pfo:.4f}) "
                f"vs SiTrack {track_index} (pt={pt_si:.4f} GeV, phi={phi_si:.4f}, theta={theta_si:.4f}): "
                f"delta_phi={delta_phi:.4f}, delta_theta={delta_theta:.4f}, "
                f"pt_pfo/pt_si={pt_ratio:.4f}"
            )

# ---------------------------------------------------------
# 5. Plot: cluster phi/theta positions (unchanged), plus the
# matched track for each PFO overlaid as red dots, so the
# track-to-cluster distances are visible directly on the plot.
# ---------------------------------------------------------

# Re-derive, for each PFO, the matched track's (phi, theta) and which
# cluster index it was matched to (so we can draw a connecting line).
pfo_track_points = []  # list of (pfo_idx, track_index, phi, theta)
pfo_track_to_cluster = []  # list of (phi_track, theta_track, phi_cluster, theta_cluster)

for pfo in range(len(pfo_energy[i])):
    track_begin = pandora_tracks_begin[i][pfo]
    track_end = pandora_tracks_end[i][pfo]

    alltracks_per_pfo = []
    for j in range(track_begin, track_end):
        track_index = pandora_tracks_index[i][j]
        collection_id = pandora_tracks_collection[i][j]
        if collection_id == 4111979324:
            alltracks_per_pfo.append(track_index)

    if len(alltracks_per_pfo) != 1:
        continue
    track_index = alltracks_per_pfo[0]

    track_info = get_track_phi_theta_p(i, track_index)
    if track_info is None:
        continue
    phi_track, theta_track, p_track = track_info
    pfo_track_points.append((pfo, track_index, phi_track, theta_track))

    cluster_begin = pandora_clusters_begin[i][pfo]
    cluster_end = pandora_clusters_end[i][pfo]
    if cluster_begin == cluster_end:
        continue
    # There is exactly one matched cluster per PFO in the well-formed case
    matched_cluster_index = pandora_clusters_index[i][cluster_begin]
    phi_c = cluster_phis[matched_cluster_index]
    theta_c = cluster_thetas[matched_cluster_index]
    pfo_track_to_cluster.append((phi_track, theta_track, phi_c, theta_c))

plt.figure(figsize=(8, 6))
plt.scatter(cluster_phis, cluster_thetas, c="black", s=30, zorder=2, label="Clusters")

for c, (phi, theta) in enumerate(zip(cluster_phis, cluster_thetas)):
    plt.annotate(
        f"C{c}\n$\\phi$={phi:.3f}\n$\\theta$={theta:.3f}",
        xy=(phi, theta),
        xytext=(5, 5),
        textcoords="offset points",
        color="red",
        fontsize=8,
    )

# Matched tracks as red dots
if pfo_track_points:
    track_phis_plot = [p[2] for p in pfo_track_points]
    track_thetas_plot = [p[3] for p in pfo_track_points]
    plt.scatter(
        track_phis_plot,
        track_thetas_plot,
        c="red",
        marker="o",
        s=40,
        zorder=3,
        label="Matched tracks",
    )
    for pfo, track_index, phi, theta in pfo_track_points:
        plt.annotate(
            f"T{track_index}\n(PFO {pfo})",
            xy=(phi, theta),
            xytext=(5, -12),
            textcoords="offset points",
            color="red",
            fontsize=7,
        )

# Dashed lines connecting each matched track to its matched cluster,
# so the deltaR "distance" is visually obvious.
for phi_t, theta_t, phi_c, theta_c in pfo_track_to_cluster:
    plt.plot(
        [phi_t, phi_c],
        [theta_t, theta_c],
        linestyle="--",
        color="red",
        linewidth=0.8,
        alpha=0.6,
        zorder=1,
    )

plt.xlabel(r"$\phi$ [rad]")
plt.ylabel(r"$\theta$ [rad]")
plt.title(f"Cluster vs Matched-Track $\\phi$/$\\theta$ Positions — Event {i}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"event_{i}_cluster_phi_theta.png", dpi=150)
plt.close()

print(f"\nSaved plot: event_{i}_cluster_phi_theta.png")