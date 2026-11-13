#!/usr/bin/env python

"""

Standalone "isolation but all" study.



For each particle type (pion, electron) this script:

  1. Finds the cluster matched to the MC particle (angular matching).

  2. Computes isolation_allhits_but for the matched cluster AND for every

     other ("unmatched") cluster in the same event.

       - iso_all  : neighbours = all nearby hits not in the cluster

       - iso_clus : neighbours = nearby hits that belong to OTHER clusters

                    (i.e. nearby hits minus unclaimed hits)

  3. Makes:

       * per particle: 2D hist of cluster transverse size vs isolation (matched)

       * pion vs electron, matched only

       * pion + electron, matched AND unmatched, all on one graph

     (each for both iso_all and iso_clus)



Usage:

    python isolation_but_all.py                 # run everything

    python isolation_but_all.py --max-events 500

"""

import argparse

import os

from types import SimpleNamespace



import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import uproot

# Corner cut: isolation fraction ~1 AND HCAL fraction ~1 (top-right of the 2D plot)
CORNER_ISO_FRAC_MIN = 0.9
CORNER_HCAL_FRAC_MIN = 0.8

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

system2name = {

    679272617: "EcalBarrelCollectionRec",

    1573202488: "HcalBarrelCollectionRec",

    3383333369: "EcalEndcapCollectionRec",

    2381985645: "HcalEndcapCollectionRec",

    3403901740: "Skip",  # muon system

}

real_systems = [

    "EcalBarrelCollectionRec",

    "HcalBarrelCollectionRec",

    "EcalEndcapCollectionRec",

    "HcalEndcapCollectionRec",

]



SYS2LOC = {

    "EcalBarrelCollectionRec": "EcalBarrel",

    "HcalBarrelCollectionRec": "HcalBarrel",

    "EcalEndcapCollectionRec": "EcalEndcap",

    "HcalEndcapCollectionRec": "HcalEndcap",

}



particles = {

    211: {

        "name": "pion",

        "color": "royalblue",

        "file": "/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root",

        "matching_bound": 0.01243161,

    },

    11: {

        "name": "electron",

        "color": "green",

        "file": "/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_11_nobib0.edm4hep.root",

        "matching_bound": 0.03639618,

    },

}



SKIP_SINGLE_CLUSTER_EVENTS = True  # same as the original (n_clusters == 1 -> skip)

METRICS = {

    "iso_all": "Isolation (all nearby hits)",

    "iso_clus": "Isolation (clustered neighbors only)",

    "iso_fraction": "Isolation fraction (clustered / all)",

}
# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_event_data(path):

    events = uproot.open(path)["events"]

    pc = events["PandoraClusters"]

    ph = events["_PandoraClusters_hits"]

    mc = events["MCParticles"]



    pos, ener, cellid = {}, {}, {}

    for name in real_systems:

        prefix = f"{name}/{name}"

        pos[name] = {ax: events[f"{prefix}.position.{ax}"].array() for ax in "xyz"}

        ener[name] = events[f"{prefix}.energy"].array()

        cellid[name] = events[f"{prefix}.cellID"].array()



    return SimpleNamespace(

        n_events=events.num_entries,

        cluster_energy=pc["PandoraClusters.energy"].array(),

        hits_begin=pc["PandoraClusters.hits_begin"].array(),

        hits_end=pc["PandoraClusters.hits_end"].array(),

        hit_index=ph["_PandoraClusters_hits.index"].array(),

        collection_id=ph["_PandoraClusters_hits.collectionID"].array(),

        pos=pos, ener=ener, cellid=cellid,

        status_mc=mc["MCParticles.generatorStatus"].array(),

        mc_x=mc["MCParticles.endpoint.x"].array(),

        mc_y=mc["MCParticles.endpoint.y"].array(),

        mc_z=mc["MCParticles.endpoint.z"].array(),

        _cache_i=None, _cache=None,

    )





def event_arrays(D, i):

    """Per-event numpy arrays for every calorimeter collection (cached)."""

    if D._cache_i != i:

        data = {}

        for name in real_systems:

            cid = np.asarray(D.cellid[name][i]).astype(np.uint64)

            data[name] = {

                "x": np.asarray(D.pos[name]["x"][i], dtype=float),

                "y": np.asarray(D.pos[name]["y"][i], dtype=float),

                "z": np.asarray(D.pos[name]["z"][i], dtype=float),

                "e": np.asarray(D.ener[name][i], dtype=float),

                "layer": ((cid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64),

                "cid": cid,

            }

        D._cache_i, D._cache = i, data

    return D._cache
# ----------------------------------------------------------------------------
# Cluster hits (all arrays are aligned with each other)
# ----------------------------------------------------------------------------

def cluster_hits(D, i, j):

    """Hits of cluster j in event i, muon hits removed, non-finite removed."""

    start, end = int(D.hits_begin[i][j]), int(D.hits_end[i][j])

    if end <= start:

        return None

    gidx = np.asarray(D.hit_index[i][start:end]).astype(np.int64)

    cids = np.asarray(D.collection_id[i][start:end])

    names = np.array([system2name.get(int(c)) for c in cids], dtype=object)

    keep = np.array([n in real_systems for n in names], dtype=bool)

    if not keep.any():

        return None

    gidx, names = gidx[keep], names[keep]



    ev = event_arrays(D, i)

    n = len(gidx)

    x, y, z, w = (np.zeros(n) for _ in range(4))

    cell = np.zeros(n, dtype=np.uint64)

    loc = np.empty(n, dtype=object)

    for name in real_systems:

        m = names == name

        if m.any():

            a, g = ev[name], gidx[m]

            x[m], y[m], z[m], w[m] = a["x"][g], a["y"][g], a["z"][g], a["e"][g]

            cell[m] = a["cid"][g]

            loc[m] = SYS2LOC[name]



    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(w)

    if not ok.any() or w[ok].sum() <= 0:

        return None

    return SimpleNamespace(sysname=names[ok], gidx=gidx[ok],

                           x=x[ok], y=y[ok], z=z[ok], w=w[ok],

                           cell=cell[ok], loc=loc[ok])





def center_of_gravity(h):

    W = h.w.sum()

    return np.array([np.sum(h.w * h.x), np.sum(h.w * h.y), np.sum(h.w * h.z)]) / W





def sigma_transverse(h):

    """sqrt of the sum of the two smaller eigenvalues of the inertia tensor."""

    pos = np.vstack([h.x, h.y, h.z]).T

    d = pos - center_of_gravity(h)

    T = np.einsum("ni,nj,n->ij", d, d, h.w) / h.w.sum()

    ev = np.linalg.eigvalsh(T)

    sigma_T = float(np.sqrt(max(ev[0] + ev[1], 0.0)))
    return sigma_T
    #if sigma_T > float(7.5):
    #    return sigma_T
    #else:
    #    return float(7.5)
    #return float(np.sqrt(max(ev[0] + ev[1], 0.0)))





def core_signal_fraction_detector(ws, cell_ids, locs):



    ws = np.asarray(ws, dtype=np.float64)

    locs = np.asarray(locs)

    ids = np.asarray(cell_ids).astype(np.uint64)



    total_energy = np.sum(ws)
# Extract layer number from CellID

    layer = (ids >> np.uint64(19)) & np.uint64(0x1FF)



    detector_masks = {

        "EcalBarrel": locs == "EcalBarrel",

        "EcalEndcap": locs == "EcalEndcap",

        "HcalBarrel": locs == "HcalBarrel",

        "HcalEndcap": locs == "HcalEndcap",

    }



    energy_total = 0.0



    count_ecal = 0

    count_hcal = 0
# Core signal energy

    energy_core_ecal = 0.0

    energy_core_hcal = 0.0
# Total energy from all hits

    energy_all_ecal = 0.0

    energy_all_hcal = 0.0



    for detector, mask in detector_masks.items():



        sub_layers = layer[mask]

        sub_ws = ws[mask]
# Number of hits

        length = len(sub_layers)



        if detector in ("EcalBarrel", "EcalEndcap"):

            count_ecal += length

            energy_all_ecal += np.sum(sub_ws)



        elif detector in ("HcalBarrel", "HcalEndcap"):

            count_hcal += length

            energy_all_hcal += np.sum(sub_ws)
# For each layer, take the largest hit energy

        for value in np.unique(sub_layers):



            layer_mask = sub_layers == value



            max_energy = np.max(sub_ws[layer_mask])



            energy_total += max_energy



            if detector in ("EcalBarrel", "EcalEndcap"):

                energy_core_ecal += max_energy



            elif detector in ("HcalBarrel", "HcalEndcap"):

                energy_core_hcal += max_energy



    total = count_ecal + count_hcal



    if total == 0 or total_energy == 0:

        return None, None, None, None, None, None



    hcal_fraction = count_hcal / total



    core_signal_fraction = energy_total / total_energy



    core_hcal_over_all = energy_core_hcal / total_energy

    core_ecal_over_all = energy_core_ecal / total_energy



    if energy_all_hcal == 0:

        core_hcal_over_hcal = 0

    else:

        core_hcal_over_hcal = energy_core_hcal / energy_all_hcal

    if energy_all_ecal == 0:

        core_ecal_over_ecal = 0

    else:

        core_ecal_over_ecal = energy_core_ecal / energy_all_ecal



    return (

        core_signal_fraction,

        hcal_fraction,

        core_hcal_over_all,

        core_ecal_over_all,

        core_hcal_over_hcal,

        core_ecal_over_ecal,

    )
# ----------------------------------------------------------------------------
# Unclaimed hits
# ----------------------------------------------------------------------------

def unclaimed_indexes(D, i, n_clusters):

    """Per-collection indices of hits in event i that belong to NO cluster."""

    claimed = {name: set() for name in real_systems}

    for j in range(n_clusters):

        start, end = int(D.hits_begin[i][j]), int(D.hits_end[i][j])

        idxs = np.asarray(D.hit_index[i][start:end])

        ids = np.asarray(D.collection_id[i][start:end])

        for sysid, idx in zip(ids, idxs):

            name = system2name.get(int(sysid))

            if name in real_systems:

                claimed[name].add(int(idx))

    return {

        name: set(range(len(D.ener[name][i]))) - claimed[name]

        for name in real_systems

    }
# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------

def cluster_matching(D, i, bound, hits_by_cluster):

    """Index of the highest-energy cluster within `bound` rad of any final-state MC particle."""

    if not hits_by_cluster:

        return None

    mask = np.asarray(D.status_mc[i]) == 1

    mx = np.asarray(D.mc_x[i])[mask]

    my = np.asarray(D.mc_y[i])[mask]

    mz = np.asarray(D.mc_z[i])[mask]

    if len(mx) == 0:

        return None

    mc_r = np.sqrt(mx**2 + my**2 + mz**2)

    good = mc_r > 0

    mx, my, mz, mc_r = mx[good], my[good], mz[good], mc_r[good]

    if len(mx) == 0:

        return None

    mc_theta = np.arccos(mz / mc_r)

    mc_phi = np.arctan2(my, mx)



    ids = list(hits_by_cluster)

    cogs = np.array([center_of_gravity(hits_by_cluster[j]) for j in ids])

    energy = np.array([hits_by_cluster[j].w.sum() for j in ids])

    c_r = np.linalg.norm(cogs, axis=1)

    good = c_r > 0

    if not good.any():

        return None

    ids = np.asarray(ids)[good]

    cogs, energy, c_r = cogs[good], energy[good], c_r[good]

    c_theta = np.arccos(cogs[:, 2] / c_r)

    c_phi = np.arctan2(cogs[:, 1], cogs[:, 0])



    cosang = (np.sin(c_theta)[:, None] * np.sin(mc_theta)[None, :]

              * np.cos(c_phi[:, None] - mc_phi[None, :])

              + np.cos(c_theta)[:, None] * np.cos(mc_theta)[None, :])

    ang = np.arccos(np.clip(cosang, -1.0, 1.0))

    cand = np.nonzero((ang <= bound).any(axis=1))[0]

    if len(cand) == 0:

        return None

    return int(ids[cand[np.argmax(energy[cand])]])
# ----------------------------------------------------------------------------
# Isolation "but all"
# ----------------------------------------------------------------------------

def isolation_allhits_but(D, i, unclaimed, h, max_R = False):

    """

    For each (collection, layer) the cluster touches (weighted by the cluster's

    energy in that layer):

      - width = energy-weighted RMS distance of the cluster's hits in that layer

        from their centroid; neighbour radius = width / 2

      - neighbours = hits in that layer NOT in this cluster, within the radius

        of any of this cluster's hits in that layer

    Returns (iso_all, iso_clus):

      iso_all  = E-weighted mean number of neighbours

      iso_clus = E-weighted mean number of neighbours that belong to some

                 other cluster (neighbours minus unclaimed neighbours)

    """


    ev = event_arrays(D, i)

    own = {name: set() for name in real_systems}

    for name, g in zip(h.sysname, h.gidx):

        own[name].add(int(g))
    # cluster energy per (collection, layer)

    layer_energy = {}

    for name, g, w in zip(h.sysname, h.gidx, h.w):

        key = (name, int(ev[name]["layer"][g]))

        layer_energy[key] = layer_energy.get(key, 0.0) + w
    num_all = den_all = num_clus = den_clus = 0.0
    for (name, lay), E_s in layer_energy.items():

        if E_s <= 0:

            continue

        a = ev[name]

        layer_idx = np.nonzero(a["layer"] == lay)[0]

        in_own = np.fromiter((g in own[name] for g in layer_idx), dtype=bool,

                             count=len(layer_idx))

        our_idx, cand_idx = layer_idx[in_own], layer_idx[~in_own]

        if len(cand_idx) == 0 or len(our_idx) == 0:

            continue



        w = a["e"][our_idx]

        W = w.sum()

        if W <= 0:

            continue

        ox, oy, oz = a["x"][our_idx], a["y"][our_idx], a["z"][our_idx]

        cx, cy, cz = np.sum(w * ox) / W, np.sum(w * oy) / W, np.sum(w * oz) / W

        width = np.sqrt(np.sum(w * ((ox - cx)**2 + (oy - cy)**2 + (oz - cz)**2)) / W)
        radius = width / 2

        
        if max_R == True:
            if radius < 6:
                radius = 6
        


        nx, ny, nz = a["x"][cand_idx], a["y"][cand_idx], a["z"][cand_idx]

        d2 = ((ox[:, None] - nx[None, :])**2

              + (oy[:, None] - ny[None, :])**2

              + (oz[:, None] - nz[None, :])**2)

        nearby_idx = cand_idx[(d2 < radius**2).any(axis=0)]

        n_neighbor = len(nearby_idx)

        n_noclus = sum(1 for g in nearby_idx if int(g) in unclaimed[name])

        num_all += E_s * n_neighbor
        den_all += E_s
        num_clus += E_s * (n_neighbor - n_noclus)
        den_clus += E_s



    iso_all = num_all / den_all if den_all > 0 else None

    iso_clus = num_clus / den_clus if den_clus > 0 else None

    return iso_all, iso_clus





def check_matched_isolation_rank(cluster_metrics, matched_id):

    """

    Check whether the MC-matched cluster has the maximum isolation

    among all valid clusters in an event.



    Parameters

    ----------

    cluster_metrics : dict

        {cluster_id: {

            "iso_all": value,

            "iso_clus": value,

            "iso_fraction": value

        }}

    matched_id : int

        Cluster ID selected by cluster_matching().



    Returns

    -------

    dict

        Whether the matched cluster is tied for the maximum of each metric.

        Invalid/non-finite values are excluded. Ties count as a maximum.

    """

    results = {}



    if matched_id not in cluster_metrics:

        return {

            "iso_all": False,

            "iso_clus": False,

            "iso_fraction": False,

        }

    for metric in ("iso_all", "iso_clus", "iso_fraction"):

        values = {

            cid: data[metric]

            for cid, data in cluster_metrics.items()

            if data[metric] is not None

            and np.isfinite(data[metric])

            and not (

                metric == "iso_fraction"

                and data[metric] < 0

            )

        }

        if matched_id not in values or not values:

            results[metric] = False

            continue

        max_value = max(values.values())
        results[metric] = np.isclose(

            values[matched_id], max_value

        )
    return results

def check_matched_min_isolation_rank(cluster_metrics, matched_id):
    """
    Check whether the MC-matched cluster is tied for the minimum value of
    each isolation metric among valid clusters in the event.

    Invalid/non-finite values are excluded. Negative iso_fraction values are
    treated as invalid, matching the maximum-isolation check.
    """
    results = {}

    if matched_id not in cluster_metrics:
        return {"iso_all": False, "iso_clus": False, "iso_fraction": False}

    for metric in ("iso_all", "iso_clus", "iso_fraction"):
        values = {
            cid: data[metric]
            for cid, data in cluster_metrics.items()
            if data[metric] is not None
            and np.isfinite(data[metric])
            and not (metric == "iso_fraction" and data[metric] < 0)
        }

        if matched_id not in values or not values:
            results[metric] = False
            continue

        min_value = min(values.values())
        results[metric] = np.isclose(values[matched_id], min_value)

    return results


# ----------------------------------------------------------------------------
# Per-particle processing
# ----------------------------------------------------------------------------

def nan_if_none(v):

    return np.nan if v is None else v





def process_particle(info, max_events=None):
    D = load_event_data(info["file"])
    n_ev = D.n_events if max_events is None else min(D.n_events, max_events)

    out = {k: [] for k in ["matched_iso_all", "matched_iso_clus", "matched_sigmaT",
                           "unmatched_iso_all", "unmatched_iso_clus", "unmatched_sigmaT",
                           "matched_hcal_frac", "unmatched_hcal_frac",
                           "matched_energy", "unmatched_energy",
                           "matched_iso_fraction", "unmatched_iso_fraction",
                           "matched_event", "unmatched_event",          
                           "matched_cluster", "unmatched_cluster",
                           "matched_nhits", "unmatched_nhits"] }    

    rank_counts = {metric: 0 for metric in ("iso_all", "iso_clus", "iso_fraction")}
    min_rank_counts = {metric: 0 for metric in ("iso_all", "iso_clus", "iso_fraction")}
    n_rank_checked = 0

    for i in range(n_ev):
        n_clusters = len(D.cluster_energy[i])

        if SKIP_SINGLE_CLUSTER_EVENTS and n_clusters == 1:
            continue

        hits_by_cluster = {}
        for j in range(n_clusters):
            h = cluster_hits(D, i, j)
            if h is not None:
                hits_by_cluster[j] = h

        matched = cluster_matching(D, i, info["matching_bound"], hits_by_cluster)
        if matched is None:
            continue

        unclaimed = unclaimed_indexes(D, i, n_clusters)
        event_metrics = {}

        for j, h in hits_by_cluster.items():
            tag = "matched" if j == matched else "unmatched"
            if sigma_transverse(h) <= 12:
                check = True
            else:
                check = False
            iso_all, iso_clus = isolation_allhits_but(D, i, unclaimed, h, check)

            if iso_all is not None and iso_clus is not None and iso_all > 0:
                iso_fraction = iso_clus / iso_all
            elif iso_all == 0:
                iso_fraction = -1
            else:
                iso_fraction = None

            out[f"{tag}_iso_fraction"].append(nan_if_none(iso_fraction))
            out[f"{tag}_iso_all"].append(nan_if_none(iso_all))
            out[f"{tag}_iso_clus"].append(nan_if_none(iso_clus))
            out[f"{tag}_sigmaT"].append(sigma_transverse(h))

            hcal_frac = core_signal_fraction_detector(h.w, h.cell, h.loc)[1]
            out[f"{tag}_hcal_frac"].append(nan_if_none(hcal_frac))
            out[f"{tag}_energy"].append(float(np.sum(h.w)))
            out[f"{tag}_event"].append(i)       
            out[f"{tag}_cluster"].append(j) 
            out[f"{tag}_nhits"].append(len(h.w))    

            event_metrics[j] = {
                "iso_all": iso_all,
                "iso_clus": iso_clus,
                "iso_fraction": iso_fraction,
            }

        rank_check = check_matched_isolation_rank(event_metrics, matched)
        min_rank_check = check_matched_min_isolation_rank(event_metrics, matched)
        n_rank_checked += 1

        for metric, is_max in rank_check.items():
            rank_counts[metric] += int(is_max)
        for metric, is_min in min_rank_check.items():
            min_rank_counts[metric] += int(is_min)

        if i % 100 == 0:
            print(f"  [{info['name']}] event {i}/{n_ev}")

    print(f"\n{info['name']}: matched-cluster isolation ranking")
    print(f"Events checked: {n_rank_checked}")
    for metric, count in rank_counts.items():
        percentage = (100.0 * count / n_rank_checked if n_rank_checked else 0.0)
        print(f"  Matched cluster has maximum {metric}: "
              f"{count}/{n_rank_checked} ({percentage:.2f}%)")

    print(f"\n{info['name']}: matched-cluster minimum-isolation ranking")   # fixed \\n -> \n
    print(f"Events checked: {n_rank_checked}")
    for metric, count in min_rank_counts.items():
        percentage = (100.0 * count / n_rank_checked if n_rank_checked else 0.0)
        print(f"  Matched cluster has minimum {metric}: "
              f"{count}/{n_rank_checked} ({percentage:.2f}%)")

    # ---- NEW: Corner cut: high isolation fraction AND high HCAL fraction ----
    for tag in ("matched", "unmatched"):
        frac = np.asarray(out[f"{tag}_iso_fraction"], dtype=float)
        hfrac = np.asarray(out[f"{tag}_hcal_frac"], dtype=float)
        evs = np.asarray(out[f"{tag}_event"])
        cls = np.asarray(out[f"{tag}_cluster"])
        en = np.asarray(out[f"{tag}_energy"], dtype=float)

        sel = (np.isfinite(frac) & np.isfinite(hfrac)
               & (frac >= CORNER_ISO_FRAC_MIN) & (hfrac >= CORNER_HCAL_FRAC_MIN))

        print(f"\n{info['name']} ({tag}) corner clusters "
              f"(iso_fraction >= {CORNER_ISO_FRAC_MIN}, "
              f"hcal_frac >= {CORNER_HCAL_FRAC_MIN}): {sel.sum()}")
        for e, c, f_, h_, E in zip(evs[sel], cls[sel], frac[sel], hfrac[sel], en[sel]):
            print(f"  event {e:5d}  cluster {c:3d}  iso_frac={f_:.3f}  "
                  f"hcal_frac={h_:.3f}  E={E:.2f}")
        if tag == "matched":
            print("  matched event numbers:", sorted(set(int(e) for e in evs[sel])))

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}

def finite(a):

    a = np.asarray(a, dtype=float)

    return a[np.isfinite(a)]





def plot_2d_vs_axis(res, info, metric, outdir):

    x, y = res["matched_sigmaT"], res[f"matched_{metric}"]

    ok = np.isfinite(x) & np.isfinite(y)

    if not ok.any():

        print(f"Skipping 2D plot {info['name']} {metric}: no data")

        return

    fig, ax = plt.subplots(figsize=(6, 5))

    h = ax.hist2d(x[ok], y[ok], bins=40, cmap="viridis", cmin=1)

    fig.colorbar(h[3], ax=ax, label="clusters")

    ax.set_xlabel(r"Cluster transverse size $\sigma_T$")

    ax.set_ylabel(METRICS[metric])

    ax.set_title(f"{info['name']}: {METRICS[metric]}\n(N = {ok.sum()})")

    fig.tight_layout()

    fig.savefig(os.path.join(outdir, f"{info['name']}_{metric}_vs_axis.png"), dpi=200)

    plt.close(fig)





def plot_hcal_frac_vs_iso(res, info, metric, outdir, tag="matched"):

    """2D hist: HCAL hit fraction (x) vs isolation value (y)."""

    x, y = res[f"{tag}_hcal_frac"], res[f"{tag}_{metric}"]

    ok = np.isfinite(x) & np.isfinite(y)

    if not ok.any():

        print(f"Skipping HCAL-frac plot {info['name']} {metric}: no data")

        return

    fig, ax = plt.subplots(figsize=(6.5, 5))

    h = ax.hist2d(x[ok], y[ok], bins=40, cmap="viridis", norm=LogNorm(), cmin=1)

    fig.colorbar(h[3], ax=ax, label="clusters (log scale)")

    ax.set_xlabel("HCAL fraction (HCAL hits / all calorimeter hits)")

    ax.set_ylabel(METRICS[metric])

    ax.set_title(f"{info['name']} ({tag}): {METRICS[metric]}\n(N = {ok.sum()})")

    fig.tight_layout()

    fig.savefig(os.path.join(outdir, f"{info['name']}_{tag}_{metric}_vs_hcalfrac.png"), dpi=200)

    plt.close(fig)





def plot_energy_vs_iso(res, info, metric, outdir, tag="matched"):

    """2D hist: cluster energy (x) vs isolation value (y)."""

    x, y = res[f"{tag}_energy"], res[f"{tag}_{metric}"]

    ok = np.isfinite(x) & np.isfinite(y)

    if not ok.any():

        print(f"Skipping energy plot {info['name']} {tag} {metric}: no data")

        return

    fig, ax = plt.subplots(figsize=(6.5, 5))

    h = ax.hist2d(x[ok], y[ok], bins=40, cmap="viridis", norm=LogNorm(), cmin=1)

    fig.colorbar(h[3], ax=ax, label="clusters (log scale)")

    ax.set_xlabel("Cluster energy (sum of hit energies)")

    ax.set_ylabel(METRICS[metric])

    ax.set_title(f"{info['name']} ({tag}): {METRICS[metric]}\n(N = {ok.sum()})")

    fig.tight_layout()

    fig.savefig(os.path.join(outdir, f"{info['name']}_{tag}_{metric}_vs_energy.png"), dpi=200)

    plt.close(fig)

def plot_2d_generic(res, info, xkey, metric, xlabel, outdir, tag="matched", bins=40):
    """2D hist: res[tag_xkey] (x) vs isolation metric (y), log colour scale.
    For iso_fraction, the -1 sentinel (iso_all == 0) gets its own bin."""
    x, y = res[f"{tag}_{xkey}"], res[f"{tag}_{metric}"]
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        print(f"Skipping {info['name']} {tag} {metric} vs {xkey}: no data")
        return

    xb = bins
    yb = bins
    n_sentinel = 0
    if metric == "iso_fraction":
        n_sentinel = int(np.sum(ok & (y < 0)))
        ymax = max(1.0, float(y[ok].max()))
        # one bin centred on -1, an empty gap, then ordinary bins from 0 to ymax
        yb = np.concatenate([[-1.1, -0.9], np.linspace(0, ymax, 41)])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    h = ax.hist2d(x[ok], y[ok], bins=[xb, yb], cmap="viridis",
                  norm=LogNorm(), cmin=1)
    fig.colorbar(h[3], ax=ax, label="clusters (log scale)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRICS[metric])

    if metric == "iso_fraction":
        ax.set_yticks([-1, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["-1 (iso_all=0)", "0", "0.25", "0.5", "0.75", "1.0"])
        ax.axhline(-0.45, color="gray", linestyle=":", linewidth=1)  # visual separator

    title = f"{info['name']} ({tag}): {METRICS[metric]}\n(N = {ok.sum()}"
    if n_sentinel:
        title += f", of which {n_sentinel} with iso_all = 0"
    ax.set_title(title + ")")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{info['name']}_{tag}_{metric}_vs_{xkey}.png"), dpi=200)
    plt.close(fig)





def common_bins(arrays, nbins=50):

    vals = np.concatenate([finite(a) for a in arrays if len(finite(a))])

    if len(vals) == 0:

        return None

    lo, hi = vals.min(), vals.max()

    if hi <= lo:

        hi = lo + 1.0

    return np.linspace(lo, hi, nbins + 1)





def plot_matched_only(all_res, metric, outdir, density=True):

    """Pion vs electron, matched clusters only."""

    bins = common_bins([r[f"matched_{metric}"] for r in all_res.values()])

    if bins is None:

        return

    plt.figure(figsize=(7, 5))

    for pdg, r in all_res.items():

        v = finite(r[f"matched_{metric}"])

        if len(v) == 0:

            continue

        plt.hist(v, bins=bins, color=particles[pdg]["color"], alpha=0.5,

                 edgecolor="black", density=density,

                 label=f"{particles[pdg]['name']} matched (N={len(v)})")

    plt.xlabel(METRICS[metric])

    plt.ylabel("Normalized counts" if density else "Counts")

    plt.title(f"Matched clusters: {METRICS[metric]}")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(os.path.join(outdir, f"matched_{metric}_pion_vs_electron.png"), dpi=150)

    plt.close()





def plot_matched_and_unmatched(all_res, metric, outdir, logy=False):

    """Raw counts vs isolation value: pion matched/unmatched, electron matched/unmatched."""

    arrays = [r[f"{tag}_{metric}"] for r in all_res.values() for tag in ("matched", "unmatched")]

    bins = common_bins(arrays)

    if bins is None:

        return

    plt.figure(figsize=(8, 5.5))

    for pdg, r in all_res.items():

        color, name = particles[pdg]["color"], particles[pdg]["name"]

        vm = finite(r[f"matched_{metric}"])

        vu = finite(r[f"unmatched_{metric}"])

        if len(vm):

            plt.hist(vm, bins=bins, color=color, alpha=0.45, edgecolor="black",

                     label=f"{name} matched (N={len(vm)})")

        if len(vu):

            plt.hist(vu, bins=bins, histtype="step", color=color, linewidth=2,

                     linestyle="--", label=f"{name} unmatched (N={len(vu)})")

    plt.xlabel(METRICS[metric])

    plt.ylabel("Counts")

    if logy:

        plt.yscale("log")

    plt.title(f"{METRICS[metric]}: matched vs unmatched clusters")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    suffix = "_logy" if logy else ""

    plt.savefig(os.path.join(outdir, f"matched_and_unmatched_{metric}{suffix}.png"), dpi=150)

    plt.close()
# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", default=".", help="where to write PNGs")

    ap.add_argument("--max-events", type=int, default=None)

    ap.add_argument("--counts", action="store_true", help="plot raw counts instead of normalized")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)



    all_res = {}

    for pdg, info in particles.items():

        print(f"Processing {info['name']} ...")

        res = process_particle(info, args.max_events)

        all_res[pdg] = res

        print(f"  {info['name']}: {len(res['matched_iso_all'])} matched, "

        f"{len(res['unmatched_iso_all'])} unmatched clusters")

        for metric in METRICS:

            plot_2d_vs_axis(res, info, metric, args.outdir)

            for tag in ("matched", "unmatched"):

                plot_hcal_frac_vs_iso(res, info, metric, args.outdir, tag=tag)

                plot_energy_vs_iso(res, info, metric, args.outdir, tag=tag)

        print("Done.")

        for tag in ("matched", "unmatched"):
            # fraction vs cluster width
            plot_2d_generic(res, info, "sigmaT", "iso_fraction",
                            r"Cluster transverse size $\sigma_T$", args.outdir, tag)
            # number of hits vs iso_all, iso_clus, iso_fraction
            for metric in METRICS:
                plot_2d_generic(res, info, "nhits", metric,
                                "Number of hits in cluster", args.outdir, tag)

    for metric in METRICS:

        plot_matched_only(all_res, metric, args.outdir, density=not args.counts)

        plot_matched_and_unmatched(all_res, metric, args.outdir)

        plot_matched_and_unmatched(all_res, metric, args.outdir, logy=True)





if __name__ == "__main__":

    main()