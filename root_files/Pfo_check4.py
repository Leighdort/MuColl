
#Imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
choices = [50]
from matplotlib.colors import LogNorm

#Pfo 
pfo_energy_over_track_pt = []
pfo_pt_over_track_pt = []
pfo_energy_values = []
pfo_pt_values = []
pfos_per_event = []
#Tracks
track_pt_values = []
track_p_values = []
tracks_per_event = []
associated_tracks_per_event = []
sitracks_per_event = []
#Cluster
cluster_energy_over_track_pt = []
cluster_energy_values = []
associated_cluster_delta_r_values = []
nearest_cluster_delta_r_values = []
associated_cluster_is_nearest = []
#General
delta_phi_values = []
delta_theta_values = []
pfo_energy_over_track_p = []
cluster_energy_over_track_p = []
pfo_track_fraction_per_event = []
closest_cluster_energy_over_track_p = []
associated_cluster_energy_over_track_p = []
mismatch_track_si = []
mismatch_track_ass = []
mismatch_event = []
#General counters
pfos_with_track_no_cluster = 0
pfos_checked = 0
pfos_bad_cluster_count = 0
pfos_bad_sitrack_match = 0
pfos_bad_track_count = 0
pfos_good = 0


#This is finding the track state associated with the calo face
def get_calo_face_state(event_i, track_index, location_target=4):
    sb = pandora_track_statesbegin[event_i][track_index]
    se = pandora_track_statesend[event_i][track_index]
    for s in range(sb, se):
        if track_states_location[event_i][s] == location_target:
            return s
    return None  # no calo-face state for this track
input_file = "reco_output.edm4hep.root"

for num in choices:
    file = uproot.open(input_file)
    events = file["events"]
    clusters = events["PandoraClusters"]
    cluster_energies = clusters["PandoraClusters.energy"].array()
    cluster_x = events["PandoraClusters/PandoraClusters.position.x"].array()
    cluster_y = events["PandoraClusters/PandoraClusters.position.y"].array()
    cluster_z = events["PandoraClusters/PandoraClusters.position.z"].array()
    pfo_energy = events["PandoraPFOs/PandoraPFOs.energy"].array()
    pfo_momx = events["PandoraPFOs/PandoraPFOs.momentum.x"].array()
    pfo_momy = events["PandoraPFOs/PandoraPFOs.momentum.y"].array()
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
    track_states_phi = events["_AllTracks_trackStates/_AllTracks_trackStates.phi"].array()
    track_states_omega = events[ "_AllTracks_trackStates/_AllTracks_trackStates.omega" ].array()
    track_states_tanLambda = events[ "_AllTracks_trackStates/_AllTracks_trackStates.tanLambda" ].array()
    track_states_x = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.x"].array()
    track_states_y = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.y"].array()
    track_states_z = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.z"].array()
    track_states_location = events["_AllTracks_trackStates/_AllTracks_trackStates.location"].array()
    sitrack_indices = events["SiTracks_objIdx/SiTracks_objIdx.index"].array()
    #Magnetic Field 
    B = 5
    

    for i in range(events.num_entries):
        # Loop over events
        num_total_tracks = len(pandora_track_statesbegin[i])
        num_total_pfos = len(pfo_energy[i])
        num_pfos_with_track = 0
        associated_track_indices = set()
        sitrack_index_set = set(sitrack_indices[i])

        for pfo in range(len(pfo_energy[i])):
            #Looping over pfos in an event
            track_begin = pandora_tracks_begin[i][pfo]
            track_end = pandora_tracks_end[i][pfo]
            has_track = False
            pT_track = None

            # ---------------------------------------------------------
            # Look for a track associated with this PFO
            # ---------------------------------------------------------
            pfos_checked += 1
            alltracks_per_pfo = []
            for j in range(track_begin, track_end):
                track_index = pandora_tracks_index[i][j]
                collection_id = pandora_tracks_collection[i][j]
                # Only look at AllTracks
                if collection_id == 4111979324:
                    alltracks_per_pfo.append(track_index)
                    has_track = True
            if len(alltracks_per_pfo) > 1:
                pfos_bad_track_count += 1
            if has_track:
                track_index = alltracks_per_pfo[0]
                if track_index not in sitrack_index_set:
                    pfos_bad_sitrack_match += 1
                associated_track_indices.add(track_index)
                state_begin = pandora_track_statesbegin[i][track_index]
                omega_track = track_states_omega[i][state_begin]
                #Getting the spatial phi, theta at the calo face
                calo_state = get_calo_face_state(i, track_index)
                if calo_state is None:
                    has_track = False
                    continue
                x = track_states_x[i][calo_state]
                y = track_states_y[i][calo_state]
                z = track_states_z[i][calo_state]
                phi_spatial_track = np.arctan2(y, x)
                theta_spatial_track = np.arctan2(np.sqrt(x**2 + y**2), z)
                # Track pT in GeV
                pT_track = abs(0.3 * B / omega_track / 1000)
                p_track = pT_track / np.sin(theta_spatial_track)
            # ---------------------------------------------------------
            # Only continue if the PFO has a track
            # ---------------------------------------------------------
            if has_track:
                num_pfos_with_track += 1
            if not has_track:
                continue
            # ---------------------------------------------------------
            # Get cluster energy associated with this PFO
            # ---------------------------------------------------------
            cluster_begin = pandora_clusters_begin[i][pfo]
            cluster_end = pandora_clusters_end[i][pfo]
            n_clusters_this_pfo = cluster_end - cluster_begin
            
            #Keeping track of pfos that don't have exactly 1 cluster
            if n_clusters_this_pfo != 1:
                pfos_bad_cluster_count += 1

            if cluster_begin == cluster_end:
                pfos_with_track_no_cluster += 1
                continue

            pfo_cluster_energy = 0.0
            for j in range(cluster_begin, cluster_end):
                cluster_index = pandora_clusters_index[i][j]
                pfo_cluster_energy += cluster_energies[i][cluster_index]

            # ---------------------------------------------------------
            # Cluster association validation
            #
            # Compare the cluster Pandora associated with this PFO to the nearest cluster in the entire event.
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

            if (
                len(alltracks_per_pfo) == 1
                and track_index in sitrack_index_set
                and n_clusters_this_pfo == 1
            ):
                pfos_good += 1

            # ---------------------------------------------------------
            # Find the nearest cluster to the track among ALL clusters in this event
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

            # Don't need to look at another AllTracks relation if the PFO should have only one track.

        tracks_per_event.append(num_total_tracks)
        pfos_per_event.append(num_total_pfos)
        num_si_tracks = len(sitrack_indices[i])
        sitracks_per_event.append(num_si_tracks)
        num_associated_tracks = len(associated_track_indices)
        associated_tracks_per_event.append(num_associated_tracks)
        pfo_track_fraction = (
            num_pfos_with_track / num_total_pfos
            if num_total_pfos > 0 else 0
        )
        pfo_track_fraction_per_event.append(pfo_track_fraction)
        missing_tracks = sitrack_index_set - associated_track_indices
        if missing_tracks:
            mismatch_track_si.append(sitrack_index_set)
            mismatch_track_ass.append(associated_track_indices)
            mismatch_event.append(i)

# NOTE: 'num' currently isn't used to pick which file gets opened above,
# so as written this only ever runs one config even if choices has more
# than one entry. sample_tag is wired to 'num' so filenames are correct
# once the file-open call is also parametrized by momentum/theta/pdgId.
sample_tag = f"211_65_p{num}"

# ---------------------------------------------------------
# Plot 1: PFO energy / track pT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(pfo_energy_over_track_pt, bins=50, histtype="step")
plt.axvline(1.0, color="k", linestyle="--", linewidth=1)
plt.xlabel(r"$E_{\mathrm{PFO}} / p_T^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("PFO Energy / Track $p_T$")
plt.grid(alpha=0.3)
plt.savefig(f"pfo_energy_over_track_pt_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 2: PFO pT / track pT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(pfo_pt_over_track_pt, bins=50, histtype="step")
plt.axvline(1.0, color="k", linestyle="--", linewidth=1)
plt.xlabel(r"$p_T^{\mathrm{PFO}} / p_T^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("PFO $p_T$ / Track $p_T$")
plt.grid(alpha=0.3)
plt.savefig(f"pfo_pt_over_track_pt_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 3: Cluster energy / track pT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(cluster_energy_over_track_pt, bins=50, histtype="step")
plt.axvline(1.0, color="k", linestyle="--", linewidth=1)
plt.xlabel(r"$E_{\mathrm{cluster}} / p_T^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("Matched Cluster Energy / Track $p_T$")
plt.grid(alpha=0.3)
plt.savefig(f"cluster_energy_over_track_pt_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 4: Cluster phi - track phi
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(delta_phi_values, bins=50, histtype="step")
plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
plt.xlabel(r"$\Delta\phi = \phi_{\mathrm{cluster}} - \phi_{\mathrm{track}}$ [rad]")
plt.ylabel("Number of charged PFOs")
plt.title(r"Matched Cluster--Track $\Delta\phi$")
plt.grid(alpha=0.3)
plt.savefig(f"cluster_track_delta_phi_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 5: Cluster theta - track theta
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(delta_theta_values, bins=50, histtype="step")
plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
plt.xlabel(r"$\Delta\theta = \theta_{\mathrm{cluster}} - \theta_{\mathrm{track}}$ [rad]")
plt.ylabel("Number of charged PFOs")
plt.title(r"Matched Cluster--Track $\Delta\theta$")
plt.grid(alpha=0.3)
plt.savefig(f"cluster_track_delta_theta_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 6: Cluster-track delta phi vs delta theta
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(delta_phi_values, delta_theta_values, bins=50, norm=LogNorm())
plt.xlabel(r"$\Delta\phi$ [rad]")
plt.ylabel(r"$\Delta\theta$ [rad]")
plt.title(r"Matched Cluster--Track Angular Differences")
plt.colorbar(label="Number of charged PFOs")
plt.savefig(f"cluster_track_delta_phi_vs_delta_theta_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 7: PFO energy / track momentum
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(pfo_energy_over_track_p, bins=50, histtype="step")
plt.axvline(1.0, color="k", linestyle="--", linewidth=1)
plt.xlabel(r"$E_{\mathrm{PFO}} / p^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("PFO Energy / Track Momentum")
plt.grid(alpha=0.3)
plt.savefig(f"pfo_energy_over_track_p_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 8: Cluster energy / track momentum
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(cluster_energy_over_track_p, bins=50, histtype="step")
plt.axvline(1.0, color="k", linestyle="--", linewidth=1)
plt.xlabel(r"$E_{\mathrm{cluster}} / p^{\mathrm{track}}$")
plt.ylabel("Number of charged PFOs")
plt.title("Cluster Energy / Track Momentum")
plt.grid(alpha=0.3)
plt.savefig(f"cluster_energy_over_track_p_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 9: Associated tracks vs PFO multiplicity
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(pfos_per_event, associated_tracks_per_event, bins=[50, 50], norm=LogNorm())
plt.xlabel("PFO multiplicity")
plt.ylabel("Number of tracks associated with PFOs")
plt.title("Associated Tracks vs PFO Multiplicity")
plt.colorbar(label="Number of events")
plt.savefig(f"associated_tracks_vs_pfo_multiplicity_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 10: Associated tracks vs track multiplicity
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(tracks_per_event, associated_tracks_per_event, bins=[50, 50], norm=LogNorm())
plt.xlabel("AllTrack Multiplicity")
plt.ylabel("Number of tracks associated with PFOs")
plt.title("Associated Tracks vs AllTrack Multiplicity")
plt.colorbar(label="Number of events")
plt.savefig(f"associated_tracks_vs_track_multiplicity_{sample_tag}.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 11: PFO-to-track association fraction per event
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist(pfo_track_fraction_per_event, bins=50, histtype="step")
plt.xlabel(r"$N_{\mathrm{PFOs\ with\ a\ track}} / N_{\mathrm{PFOs}}$")
plt.ylabel("Number of events")
plt.title("PFO-to-Track Association Fraction")
plt.xlim(0, 1)
plt.grid(alpha=0.3)
plt.savefig(f"pfo_track_association_fraction_per_event_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 12: Cluster energy vs track pT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(track_pt_values, cluster_energy_values, bins=[50, 50], norm=LogNorm())
plt.xlabel(r"$p_T^{\mathrm{track}}$ [GeV]")
plt.ylabel(r"$E_{\mathrm{cluster}}$ [GeV]")
plt.title("Matched Cluster Energy vs Track $p_T$")
plt.colorbar(label="Number of charged PFOs")
plt.savefig(f"cluster_energy_vs_track_pt_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 13: Cluster energy vs track momentum
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(track_p_values, cluster_energy_values, bins=[50, 50], norm=LogNorm())
plt.xlabel(r"$p^{\mathrm{track}}$ [GeV]")
plt.ylabel(r"$E_{\mathrm{cluster}}$ [GeV]")
plt.title("Matched Cluster Energy vs Track $p$")
plt.colorbar(label="Number of charged PFOs")
plt.savefig(f"cluster_energy_vs_track_p_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 14: Pandora-associated cluster deltaR vs nearest cluster deltaR
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.hist2d(
    nearest_cluster_delta_r_values,
    associated_cluster_delta_r_values,
    bins=50,
    norm=LogNorm()
)

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
plt.savefig(f"associated_vs_nearest_cluster_deltaR_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 15: Pandora-associated cluster vs nearest cluster
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
n_not_nearest = np.sum(~np.array(associated_cluster_is_nearest))
n_nearest = np.sum(associated_cluster_is_nearest)

plt.bar(
    ["Not nearest", "Nearest"],
    [n_not_nearest, n_nearest]
)
plt.xlabel("Is Pandora-associated cluster the nearest cluster?")
plt.ylabel("Number of charged PFOs")
plt.title("Track--Cluster Association")
plt.grid(alpha=0.3, axis="y")
plt.savefig(f"pandora_cluster_association_{sample_tag}.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# Shared axis limit for deltaR scatter plots
# ---------------------------------------------------------

# Use the 99th percentile to reduce the effect of large deltaR outliers
# and keep the main distribution visible.

delta_r_scatter_limit = np.percentile(
    np.concatenate([
        nearest_cluster_delta_r_values,
        associated_cluster_delta_r_values
    ]),
    99
)


# ---------------------------------------------------------
# Plot 16: Pandora-associated cluster deltaR vs nearest cluster deltaR
# Color = matched cluster E / track p
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
    label=r"$E_{\mathrm{cluster\ (matched)}} / p_{\mathrm{track}}$"
)
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, delta_r_scatter_limit)
plt.ylim(0, delta_r_scatter_limit)
plt.savefig(
    f"associated_vs_nearest_cluster_deltaR_colored_energy_{sample_tag}.png",
    dpi=150
)
plt.close()


# ---------------------------------------------------------
# Plot 17: Pandora-associated cluster deltaR vs nearest cluster deltaR
# Color = closest cluster E / track p
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
    label=r"$E_{\mathrm{cluster\ (closest)}} / p_{\mathrm{track}}$"
)
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, delta_r_scatter_limit)
plt.ylim(0, delta_r_scatter_limit)
plt.savefig(
    f"associated_vs_nearest_cluster_deltaR_colored_closest_energy_{sample_tag}.png",
    dpi=150
)
plt.close()


# ---------------------------------------------------------
# Plot 18: Which cluster has the better E/p match?
# ---------------------------------------------------------

# Better match = cluster whose E/p is closer to 1.

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
    s=10,
    alpha=0.5,
    label="Pandora cluster: better $E/p$ match"
)
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[closest_better],
    np.array(associated_cluster_delta_r_values)[closest_better],
    s=10,
    alpha=0.5,
    label="Nearest cluster: better $E/p$ match"
)
plt.scatter(
    np.array(nearest_cluster_delta_r_values)[ties],
    np.array(associated_cluster_delta_r_values)[ties],
    s=10,
    alpha=0.5,
    label="Tie"
)

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
plt.xlim(0, delta_r_scatter_limit)
plt.ylim(0, delta_r_scatter_limit)
plt.savefig(
    f"associated_vs_nearest_cluster_energy_match_{sample_tag}.png",
    dpi=150
)
plt.close()


# ---------------------------------------------------------
# Plot 19: SiTracks vs PFO multiplicity
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
plt.savefig(f"sitracks_vs_pfo_multiplicity_{sample_tag}.png", dpi=150)
plt.close()
print("\n-----------------------------------------")
print("PFO Structure Validation")
print("-----------------------------------------")
print(f"PFOs checked: {pfos_checked}")
print(f"PFOs with != 1 AllTracks track: {pfos_bad_track_count}")
print(f"PFOs with != 1 cluster: {pfos_bad_cluster_count}")
print(f"PFOs with a track but no cluster at all: {pfos_with_track_no_cluster}")
print(f"PFOs where AllTracks != SiTrack: {pfos_bad_sitrack_match}")
print(f"PFOs passing all checks: {pfos_good}")
print("-----------------------------------------")

print("\n-----------------------------------------")
print("Extra SI Tracks: ")
print("-----------------------------------------")
print(f"Number of Events with SI Tracks without being matched: {len(mismatch_event)}")
for i, index in enumerate(mismatch_event):
    print(f"Mismatched Event: {index}")
    print(f"SI tracks: {sorted(int(x) for x in mismatch_track_si[i])}")
    print(f"Associated tracks: {sorted(int(x) for x in mismatch_track_ass[i])}")
print("-----------------------------------------")

print("\n-----------------------------------------")
print("Cluster association validation")
print("-----------------------------------------")
num_tracks_with_cluster = len(associated_cluster_is_nearest)
num_associated_is_nearest = np.sum(associated_cluster_is_nearest)

if num_tracks_with_cluster > 0:
    cluster_association_fraction = (
        num_associated_is_nearest / num_tracks_with_cluster
    )
    print(f"Tracks with a cluster: {num_tracks_with_cluster}")
    print(f"Pandora cluster is nearest: {num_associated_is_nearest}")
    print(f"Association fraction: {cluster_association_fraction:.4f}")
    print(f"Association fraction (%): {100 * cluster_association_fraction:.2f}%")
print("-----------------------------------------")

print("\n-----------------------------------------")
print("Cluster / Track Energy Correlation")
print("-----------------------------------------")
print("Pandora Cluster E/p is closer to 1:", np.sum(pandora_better))
print("Closest Cluster E/p is closer to 1:", np.sum(closest_better))
print("Pandora & Closest Cluster E/p is a Tie:", np.sum(ties))
print("Total:", len(ties))
print("Pandora Cluster is the closest cluster among ties:", np.sum(same_clus))
print("-----------------------------------------")


print("\n-----------------------------------------")
print("Energy / Momentum Ratio Summary Statistics")
print("-----------------------------------------")

def print_ratio_stats(name, values):
    values = np.asarray(values)
    if len(values) == 0:
        print(f"{name}: no entries")
        return
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    print(
        f"{name}: "
        f"mean={mean:.4f}, median={median:.4f}, "
        f"std={std:.4f}, IQR={iqr:.4f}, n={len(values)}"
    )

print_ratio_stats("Matched cluster E / track p", cluster_energy_over_track_p)
print_ratio_stats("PFO energy / track p", pfo_energy_over_track_p)
print_ratio_stats("PFO energy / track pT", pfo_energy_over_track_pt)
print_ratio_stats("PFO pT / track pT", pfo_pt_over_track_pt)
print("-----------------------------------------")