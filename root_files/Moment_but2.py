#!/usr/bin/env python3
"""
Compare two isolation definitions for pion and electron Pandora clusters.

The script calculates both definitions in one event loop:
  * dynamic radius: radius = energy-weighted RMS cluster width / 2
  * fixed radius:   radius = 7.5

For each definition it records matched and unmatched clusters, prints ranking
and corner-cut summaries, and produces:
  * 1D distributions overlaid for dynamic vs fixed radius
  * 2D distributions side-by-side with shared axis limits for a fair comparison
  * the original-style pion/electron and matched/unmatched summary plots

Run:
    python compare_isolation_outputs.py
    python compare_isolation_outputs.py --max-events 500
    python compare_isolation_outputs.py --outdir isolation_comparison --counts
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORNER_ISO_FRAC_MIN = 0.9
CORNER_HCAL_FRAC_MIN = 0.8
SKIP_SINGLE_CLUSTER_EVENTS = True
#FIXED_RADIUS = 7.5
#MAX_SIGMA_T = 15.0
MAX_SIGMA_T = 12
FIXED_RADIUS = 6

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
METRICS = {
    "iso_all": "Isolation (all nearby hits)",
    "iso_clus": "Isolation (clustered neighbors only)",
    "iso_fraction": "Isolation fraction (clustered / all)",
}
TAGS = ("matched", "unmatched")
RADIUS_MODES = ("dynamic", "fixed")
PLOT_COLORS = {
    11: {  # electron
        "fixed": "green",
        "dynamic": "red",
    },
    211: {  # pion
        "fixed": "blue",
        "dynamic": "orange",
    },
}
X_VARIABLES = {
    "sigmaT": (r"Cluster transverse size $\sigma_T$", "Cluster transverse size"),
    "hcal_frac": ("HCAL fraction (HCAL hits / all calorimeter hits)", "HCAL fraction"),
    "energy": ("Cluster energy (sum of hit energies)", "Cluster energy"),
    "nhits": ("Number of hits in cluster", "Number of hits"),
}


# ---------------------------------------------------------------------------
# Loading and event arrays
# ---------------------------------------------------------------------------

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
    """Per-event numpy arrays for each calorimeter collection (cached)."""
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


# ---------------------------------------------------------------------------
# Cluster hit handling and derived quantities
# ---------------------------------------------------------------------------

def cluster_hits(D, i, j):
    """Hits of cluster j in event i; omit muon hits and non-finite coordinates."""
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
    x, y, z, w = (np.zeros(n, dtype=float) for _ in range(4))
    cell = np.zeros(n, dtype=np.uint64)
    loc = np.empty(n, dtype=object)

    for name in real_systems:
        m = names == name
        if m.any():
            a, g = ev[name], gidx[m]
            valid_idx = (g >= 0) & (g < len(a["e"]))
            # Invalid references should not be used to index detector arrays.
            positions = np.flatnonzero(m)
            if not valid_idx.all():
                names[positions[~valid_idx]] = None
                continue
            x[m], y[m], z[m], w[m] = a["x"][g], a["y"][g], a["z"][g], a["e"][g]
            cell[m] = a["cid"][g]
            loc[m] = SYS2LOC[name]

    ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(w)
          & np.array([n in real_systems for n in names]))
    if not ok.any() or np.sum(w[ok]) <= 0:
        return None
    return SimpleNamespace(sysname=names[ok], gidx=gidx[ok],
                           x=x[ok], y=y[ok], z=z[ok], w=w[ok],
                           cell=cell[ok], loc=loc[ok])


def center_of_gravity(h):
    W = h.w.sum()
    return np.array([np.sum(h.w * h.x), np.sum(h.w * h.y), np.sum(h.w * h.z)]) / W


def sigma_transverse(h):
    """sqrt of the sum of the two smaller eigenvalues of the weighted inertia tensor."""
    pos = np.vstack([h.x, h.y, h.z]).T
    d = pos - center_of_gravity(h)
    T = np.einsum("ni,nj,n->ij", d, d, h.w) / h.w.sum()
    ev = np.linalg.eigvalsh(T)
    return float(np.sqrt(max(ev[0] + ev[1], 0.0)))

def moment_difference(results, outdir, particle):
    dynamic = results["dynamic"]
    fixed = results["fixed"]

    for tag in TAGS:
        for moment in ["iso_all", "iso_clus", "iso_fraction"]:
            difference_ratio = []

            x_dynamic = dynamic[f"{tag}_{moment}"]
            y_fixed = fixed[f"{tag}_{moment}"]

            valid = (
                np.isfinite(x_dynamic)
                & np.isfinite(y_fixed)
                & (y_fixed != 0)
            )

            difference_ratio = x_dynamic[valid] / y_fixed[valid]

            # Keep only finite ratios
            difference_ratio = difference_ratio[np.isfinite(difference_ratio)]

            if len(difference_ratio) == 0:
                continue

            # Histogram and Gaussian fit
            mu = np.mean(difference_ratio)
            sigma = np.std(difference_ratio)

            if sigma == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))

            counts, bins, _ = ax.hist(
                difference_ratio,
                bins=50,
                density=True,
                alpha=0.6,
                label="Dynamic / Fixed",
            )

            x = np.linspace(bins[0], bins[-1], 500)
            gaussian = (
                1 / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            )

            '''
            ax.plot(
                x,
                gaussian,
                linewidth=2,
                label=fr"Gaussian: $\mu={mu:.3f},\ \sigma={sigma:.3f}$",
            )
            '''

            ax.set_xlabel(f"{moment} ratio (dynamic / fixed)")
            ax.set_ylabel("Probability density")
            ax.set_title(f"{tag.capitalize()} clusters: {moment}")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.35)

            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"ratio_{tag}_{moment}_{particle}.pdf")
            )
            plt.close(fig)
    
def core_signal_fraction_detector(ws, cell_ids, locs):
    """Return core-energy fraction, HCAL hit fraction, and related core fractions."""
    ws = np.asarray(ws, dtype=np.float64)
    locs = np.asarray(locs)
    ids = np.asarray(cell_ids).astype(np.uint64)
    total_energy = np.sum(ws)
    layer = (ids >> np.uint64(19)) & np.uint64(0x1FF)
    detector_masks = {
        "EcalBarrel": locs == "EcalBarrel",
        "EcalEndcap": locs == "EcalEndcap",
        "HcalBarrel": locs == "HcalBarrel",
        "HcalEndcap": locs == "HcalEndcap",
    }
    energy_total = energy_core_ecal = energy_core_hcal = 0.0
    energy_all_ecal = energy_all_hcal = 0.0
    count_ecal = count_hcal = 0

    for detector, mask in detector_masks.items():
        sub_layers, sub_ws = layer[mask], ws[mask]
        if detector.startswith("Ecal"):
            count_ecal += len(sub_layers)
            energy_all_ecal += np.sum(sub_ws)
        else:
            count_hcal += len(sub_layers)
            energy_all_hcal += np.sum(sub_ws)
        for value in np.unique(sub_layers):
            layer_mask = sub_layers == value
            if not np.any(layer_mask):
                continue
            max_energy = np.max(sub_ws[layer_mask])
            energy_total += max_energy
            if detector.startswith("Ecal"):
                energy_core_ecal += max_energy
            else:
                energy_core_hcal += max_energy

    total_hits = count_ecal + count_hcal
    if total_hits == 0 or total_energy == 0:
        return (None,) * 6

    hcal_fraction = count_hcal / total_hits
    core_signal_fraction = energy_total / total_energy
    core_hcal_over_all = energy_core_hcal / total_energy
    core_ecal_over_all = energy_core_ecal / total_energy
    core_hcal_over_hcal = energy_core_hcal / energy_all_hcal if energy_all_hcal else 0.0
    core_ecal_over_ecal = energy_core_ecal / energy_all_ecal if energy_all_ecal else 0.0
    return (core_signal_fraction, hcal_fraction, core_hcal_over_all,
            core_ecal_over_all, core_hcal_over_hcal, core_ecal_over_ecal)

def unclaimed_indexes(D, i, n_clusters):
    """Per-collection hit indices belonging to no cluster."""
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

# ---------------------------------------------------------------------------
# Matching and isolation
# ---------------------------------------------------------------------------

def cluster_matching(D, i, bound, hits_by_cluster):
    """Highest-energy cluster within bound radians of any final-state MC particle."""
    if not hits_by_cluster:
        return None
    mask = np.asarray(D.status_mc[i]) == 1
    mx, my, mz = (np.asarray(v)[mask] for v in (D.mc_x[i], D.mc_y[i], D.mc_z[i]))
    if len(mx) == 0:
        return None
    mc_r = np.sqrt(mx**2 + my**2 + mz**2)
    good = mc_r > 0
    mx, my, mz, mc_r = mx[good], my[good], mz[good], mc_r[good]
    if len(mx) == 0:
        return None
    mc_theta = np.arccos(np.clip(mz / mc_r, -1.0, 1.0))
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
    c_theta = np.arccos(np.clip(cogs[:, 2] / c_r, -1.0, 1.0))
    c_phi = np.arctan2(cogs[:, 1], cogs[:, 0])

    cosang = (np.sin(c_theta)[:, None] * np.sin(mc_theta)[None, :]
              * np.cos(c_phi[:, None] - mc_phi[None, :])
              + np.cos(c_theta)[:, None] * np.cos(mc_theta)[None, :])
    ang = np.arccos(np.clip(cosang, -1.0, 1.0))
    cand = np.nonzero((ang <= bound).any(axis=1))[0]
    if len(cand) == 0:
        return None
    return int(ids[cand[np.argmax(energy[cand])]])


def isolation_allhits_but(D, i, unclaimed, h, fixed_radius=False):
    """
    Return (iso_all, iso_clus), energy-weighted over collection/layer.

    Dynamic mode uses radius = width/2, where width is the cluster-hit energy-
    weighted RMS distance from the centroid within that layer.
    Fixed mode uses radius = 7.5.
    """
    num_all = den_all = num_clus = den_clus = 0.0
    e_near = 0.0                      # NEW
    num_rad = den_rad = 0.0           # NEW
    ev = event_arrays(D, i)
    own = {name: set() for name in real_systems}
    for name, g in zip(h.sysname, h.gidx):
        own[name].add(int(g))

    layer_energy = {}
    for name, g, w in zip(h.sysname, h.gidx, h.w):
        key = (name, int(ev[name]["layer"][g]))
        layer_energy[key] = layer_energy.get(key, 0.0) + float(w)

    num_all = den_all = num_clus = den_clus = 0.0
    for (name, lay), E_s in layer_energy.items():
        if E_s <= 0:
            continue
        a = ev[name]
        layer_idx = np.nonzero(a["layer"] == lay)[0]
        in_own = np.fromiter((g in own[name] for g in layer_idx),
                             dtype=bool, count=len(layer_idx))
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
        radius = FIXED_RADIUS if fixed_radius else width / 2.0
        radius = max(width / 2.0, FIXED_RADIUS) if fixed_radius else width / 2.0

        nx, ny, nz = a["x"][cand_idx], a["y"][cand_idx], a["z"][cand_idx]
        d2 = ((ox[:, None] - nx[None, :])**2
              + (oy[:, None] - ny[None, :])**2
              + (oz[:, None] - nz[None, :])**2)
        nearby_idx = cand_idx[(d2 < radius**2).any(axis=0)]
        n_neighbor = len(nearby_idx)
        n_unclaimed = sum(int(g) in unclaimed[name] for g in nearby_idx)
        e_near += float(a["e"][nearby_idx].sum())     # NEW
        num_rad += E_s * radius                       # NEW
        den_rad += E_s                                # NEW
        num_all += E_s * n_neighbor
        den_all += E_s
        num_clus += E_s * (n_neighbor - n_unclaimed)
        den_clus += E_s

    iso_all = num_all / den_all if den_all > 0 else None
    iso_clus = num_clus / den_clus if den_clus > 0 else None
    mean_radius = num_rad / den_rad if den_rad > 0 else None
    e_near = e_near if den_all > 0 else None
    return iso_all, iso_clus, e_near, mean_radius


def check_rank(cluster_metrics, matched_id, want_max=True):
    results = {}
    if matched_id not in cluster_metrics:
        return {m: False for m in METRICS}
    for metric in METRICS:
        vals = {
            cid: data[metric] for cid, data in cluster_metrics.items()
            if data[metric] is not None and np.isfinite(data[metric])
            and not (metric == "iso_fraction" and data[metric] < 0)
        }
        if matched_id not in vals or not vals:
            results[metric] = False
        else:
            target = max(vals.values()) if want_max else min(vals.values())
            results[metric] = bool(np.isclose(vals[matched_id], target))
    return results


def nan_if_none(value):
    return np.nan if value is None else value


# ---------------------------------------------------------------------------
# Process both radius definitions in one pass
# ---------------------------------------------------------------------------

def empty_result():
    keys = []
    for tag in TAGS:
        keys += [f"{tag}_{k}" for k in (
            "iso_all", "iso_clus", "sigmaT", "hcal_frac", "energy",
            "iso_fraction", "event", "cluster", "nhits",
            "nb_energy", "radius"
        )]
    return {key: [] for key in keys}


def unchanged_summary(all_results):
    """Fraction of clusters/events whose value is identical for dynamic and fixed radius."""
    for pdg, modes in all_results.items():
        name = particles[pdg]["name"]
        dyn, fix = modes["dynamic"], modes["fixed"]
        print(f"\n{name}: entries unchanged between dynamic and fixed radius")
        for metric in METRICS:
            ev_all, valid_all, same_all = [], [], []
            for tag in TAGS:
                a = np.asarray(dyn[f"{tag}_{metric}"], dtype=float)
                b = np.asarray(fix[f"{tag}_{metric}"], dtype=float)
                valid = np.isfinite(a) & np.isfinite(b)
                same = valid & np.isclose(a, b, rtol=0.0, atol=1e-12)
                n_valid, n_same = int(valid.sum()), int(same.sum())
                print(f"  {metric:13s} {tag:9s} clusters: {n_same}/{n_valid} "
                      f"({100*n_same/n_valid if n_valid else 0:.2f}%)")
                ev_all.append(np.asarray(dyn[f"{tag}_event"], dtype=float))
                valid_all.append(valid)
                same_all.append(same)

            ev = np.concatenate(ev_all)
            valid = np.concatenate(valid_all)
            same = np.concatenate(same_all)
            if not valid.any():
                continue
            ev_u, inv = np.unique(ev[valid], return_inverse=True)
            n_changed = np.bincount(inv, weights=(~same[valid]).astype(float))
            n_ev = len(ev_u)
            n_ev_same = int(np.sum(n_changed == 0))
            print(f"  {metric:13s} events (all clusters unchanged): {n_ev_same}/{n_ev} "
                  f"({100*n_ev_same/n_ev:.2f}%)")


def plot_radius_vs_isofrac(all_results, outdir):
    """Dynamic radius (before replacement by 7.5) vs iso_fraction, for both modes."""
    for pdg, modes in all_results.items():
        name = particles[pdg]["name"]
        for tag in TAGS:
            x = np.asarray(modes["dynamic"][f"{tag}_radius"], dtype=float)
            ys = {m: np.asarray(modes[m][f"{tag}_iso_fraction"], dtype=float)
                  for m in RADIUS_MODES}
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True,
                                     constrained_layout=True)
            meshes, max_count = [], 0.0
            finite_x = x[np.isfinite(x)]
            xedges = np.linspace(finite_x.min(), finite_x.max() + 1e-9, 41)
            # One strip for the -1 sentinel, then 20 bins over [0, 1]
            yedges = np.concatenate([[-1.1, -0.9], np.linspace(0.0, 1.0, 21)])
            for ax, mode in zip(axes, RADIUS_MODES):
                y = ys[mode]
                valid = np.isfinite(x) & np.isfinite(y)          # keep -1
                n_iso = int(np.sum(valid & (y < 0)))             # fully isolated
                if valid.any():
                    h = ax.hist2d(x[valid], y[valid], bins=[xedges, yedges],
                                cmap="viridis", cmin=1)
                    meshes.append(h[3])
                    max_count = max(max_count, float(np.nanmax(h[0])))
                ax.set_xlim(xedges[0], xedges[-1])
                ax.set_ylim(yedges[0], yedges[-1])
                ax.set_yticks([-1, 0, 0.5, 1])
                ax.set_xlabel("Dynamic radius (energy-weighted mean of width/2)")
                ax.set_ylabel(METRICS["iso_fraction"])
                ax.set_title(f"{mode.capitalize()} (N={int(valid.sum())}, "
                            f"iso_all=0 [y=-1]: {n_iso})")

            # One shared color scale and a single color bar on the right
            if meshes:
                for mesh in meshes:
                    mesh.set_clim(1, max(1, max_count))
                fig.colorbar(meshes[0], ax=axes, location="right", label="Clusters")

            fig.suptitle(f"{name} {tag}: dynamic radius vs isolation fraction")
            fig.savefig(os.path.join(outdir, f"radius_vs_isofrac_{name}_{tag}.png"), dpi=180)
            plt.close(fig)


def plot_added_energy_fraction(all_results, outdir, nbins=60):
    """(E_nearby_fixed - E_nearby_dynamic) / E_nearby_dynamic per cluster."""
    for pdg, modes in all_results.items():
        name = particles[pdg]["name"]
        print(f"\n{name}: energy added by fixed radius / dynamic nearby energy")

        # Compute fractions for both tags first so they can share bins
        fracs = {}
        for tag in TAGS:
            e_dyn = np.asarray(modes["dynamic"][f"{tag}_nb_energy"], dtype=float)
            e_fix = np.asarray(modes["fixed"][f"{tag}_nb_energy"], dtype=float)
            valid = np.isfinite(e_dyn) & np.isfinite(e_fix) & (e_dyn > 0)
            if not valid.any():
                continue
            fracs[tag] = (e_fix[valid] - e_dyn[valid]) / e_dyn[valid]

        if not fracs:
            continue
        # Shared bin edges across matched and unmatched
        bins = common_bins(list(fracs.values()), nbins=nbins)
        fig, ax = plt.subplots(figsize=(9, 6))
        for tag, frac in fracs.items():
            print(f"  {tag:9s} N={len(frac)} mean={frac.mean():.4f} "
                  f"median={np.median(frac):.4f} "
                  f"zero-added={np.mean(frac == 0)*100:.1f}% "
                  f"negative={np.mean(frac < 0)*100:.1f}%")
            ax.hist(frac, bins=bins, histtype="step", linewidth=1.8,
                    linestyle="-" if tag == "matched" else "--",
                    label=f"{tag} (N={len(frac)})")

        ax.set_yscale("log")
        ax.set_xlabel(r"$(E_{\rm nearby}^{\rm fixed} - E_{\rm nearby}^{\rm dyn}) / E_{\rm nearby}^{\rm dyn}$")
        ax.set_ylabel("Counts")
        ax.set_title(f"Energy added by fixed radius = {FIXED_RADIUS} ({name})")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"added_energy_fraction_{name}.png"), dpi=180)
        plt.close(fig)

def process_particle(info, max_events=None):
    D = load_event_data(info["file"])
    n_ev = D.n_events if max_events is None else min(D.n_events, max_events)
    results = {mode: empty_result() for mode in RADIUS_MODES}
    rank_stats = {
        mode: {
            "checked": 0,
            "max": {m: 0 for m in METRICS},
            "min": {m: 0 for m in METRICS},
        } for mode in RADIUS_MODES
    }

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
        cat = True
        # The geometry and event matching are shared; only isolation radius differs.
        per_mode_metrics = {mode: {} for mode in RADIUS_MODES}
        for j, h in hits_by_cluster.items():
            sigma_t = sigma_transverse(h)
            if sigma_t <= MAX_SIGMA_T:
                tag = "matched" if j == matched else "unmatched"
                hcal_frac = core_signal_fraction_detector(h.w, h.cell, h.loc)[1]
                common = {
                    "sigmaT": sigma_t,
                    "hcal_frac": nan_if_none(hcal_frac),
                    "energy": float(np.sum(h.w)),
                    "event": i,
                    "cluster": j,
                    "nhits": len(h.w),
                }
                for mode in RADIUS_MODES:
                    iso_all, iso_clus, e_near, mean_radius = isolation_allhits_but(
                        D, i, unclaimed, h, fixed_radius=(mode == "fixed")
                    )
                    if iso_all == 0:
                        #print("meow")
                        iso_fraction = -1.0
                    elif iso_all is not None and iso_clus is not None and iso_all > 0:
                        iso_fraction = iso_clus / iso_all
                    else:
                        iso_fraction = None
                    row = {
                        **common,
                        "iso_all": nan_if_none(iso_all),
                        "iso_clus": nan_if_none(iso_clus),
                        "iso_fraction": nan_if_none(iso_fraction),
                        "nb_energy": nan_if_none(e_near),
                        "radius": nan_if_none(mean_radius)
                    }
                    for key, value in row.items():
                        results[mode][f"{tag}_{key}"].append(value)
                    per_mode_metrics[mode][j] = {
                        "iso_all": iso_all, "iso_clus": iso_clus,
                        "iso_fraction": iso_fraction,
                    }

        for mode in RADIUS_MODES:
            rank_stats[mode]["checked"] += 1
            for metric, val in check_rank(per_mode_metrics[mode], matched, True).items():
                rank_stats[mode]["max"][metric] += int(val)
            for metric, val in check_rank(per_mode_metrics[mode], matched, False).items():
                rank_stats[mode]["min"][metric] += int(val)

        if i % 100 == 0:
            print(f"  [{info['name']}] event {i}/{n_ev}")

    for mode in RADIUS_MODES:
        print(f"\n{info['name']} — {mode} radius")
        checked = rank_stats[mode]["checked"]
        print(f"Events checked for ranking: {checked}")
        for metric in METRICS:
            mx = rank_stats[mode]["max"][metric]
            mn = rank_stats[mode]["min"][metric]
            print(f"  Matched cluster maximum {metric}: {mx}/{checked} "
                  f"({100*mx/checked if checked else 0:.2f}%)")
            print(f"  Matched cluster minimum {metric}: {mn}/{checked} "
                  f"({100*mn/checked if checked else 0:.2f}%)")
        #print_corner_clusters(info, results[mode])

    return {
        mode: {k: np.asarray(v, dtype=float) for k, v in data.items()}
        for mode, data in results.items()
    }




'''
def print_corner_clusters(info, res):
    for tag in TAGS:
        frac = res[f"{tag}_iso_fraction"]
        frac = np.asarray(frac, dtype=float)
        hfrac = res[f"{tag}_hcal_frac"]
        hfrac = np.asarray(gfrac, dtype=float
        evs = res[f"{tag}_event"]
        cls = res[f"{tag}_cluster"]
        en = res[f"{tag}_energy"]
        sel = (np.isfinite(frac) & np.isfinite(hfrac)
               & (frac >= CORNER_ISO_FRAC_MIN)
               & (hfrac >= CORNER_HCAL_FRAC_MIN))
        print(f"  {tag} corner clusters (iso_fraction >= {CORNER_ISO_FRAC_MIN}, "
              f"hcal_frac >= {CORNER_HCAL_FRAC_MIN}): {int(sel.sum())}")
        for e, c, f, hf, E in zip(evs[sel], cls[sel], frac[sel], hfrac[sel], en[sel]):
            print(f"    event {int(e):5d} cluster {int(c):3d} "
                  f"iso_frac={f:.3f} hcal_frac={hf:.3f} E={E:.2f}")
        if tag == "matched":
            print("    matched event numbers:", sorted(set(int(e) for e in evs[sel])))
'''

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def common_bins(arrays, nbins=50):
    cleaned = [finite(a) for a in arrays]
    cleaned = [a for a in cleaned if len(a)]
    if not cleaned:
        return None
    vals = np.concatenate(cleaned)
    lo, hi = np.min(vals), np.max(vals)
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, nbins + 1)



def plot_1d_comparison(all_results, metric, outdir, counts=False):
    arrays = [
        data[mode][f"{tag}_{metric}"]
        for data in all_results.values()
        for mode in RADIUS_MODES
        for tag in TAGS
    ]
    bins = common_bins(arrays)
    if bins is None:
        print(f"Skipping 1D {metric}: no finite values")
        return
    for pdg, data in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        name = particles[pdg]["name"]

        for mode in RADIUS_MODES:
            color = PLOT_COLORS[pdg][mode]

            for tag in TAGS:
                vals = finite(data[mode][f"{tag}_{metric}"])
                '''
                if metric == "iso_fraction":
                    vals = vals[vals >= 0]
                '''

                if not len(vals):
                    continue

                # Solid = matched; dashed = unmatched
                linestyle = "-" if tag == "matched" else "--"

                ax.hist(
                    vals,
                    bins=bins,
                    histtype="step",
                    linewidth=1.8,
                    color=color,
                    linestyle=linestyle,
                    density=not counts,
                    label=(
                        f"{mode.capitalize()}, {tag} "
                        f"(N={len(vals)})"
                    ),
                )

        ax.set_xlabel(METRICS[metric])
        ax.set_ylabel("Counts" if counts else "Normalized density")
        ax.set_title(
            f"{METRICS[metric]}: dynamic vs fixed radius ({name})"
        )
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=9)
        fig.tight_layout()

        fig.savefig(
            os.path.join(outdir, f"compare_1d_{metric}_{name}.png"),
            dpi=180,
        )
        plt.close(fig)



def plot_2d_pair(ax, res, tag, xkey, metric, xlabel, title, bins=40):
    x = np.asarray(res[f"{tag}_{xkey}"], dtype=float)
    y = np.asarray(res[f"{tag}_{metric}"], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if metric == "iso_fraction":
        valid
    if not np.any(valid):
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRICS[metric])
        return None
    hist = ax.hist2d(x[valid], y[valid], bins=bins, cmap="viridis", cmin=1)
    ax.set_title(f"{title} (N={int(valid.sum())})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRICS[metric])
    return hist[3]


def plot_2d_comparisons(all_results, outdir):
    """Side-by-side dynamic/fixed plots; use common limits and shared bin edges."""
    for pdg, modes in all_results.items():
        name = particles[pdg]["name"]
        for tag in TAGS:
            for xkey, (xlabel, _) in X_VARIABLES.items():
                for metric in METRICS:
                    x_arrays, y_arrays = [], []
                    for mode in RADIUS_MODES:
                        x = finite(modes[mode][f"{tag}_{xkey}"])
                        y = finite(modes[mode][f"{tag}_{metric}"])
                        valid_len = min(len(x), len(y))
                        if valid_len:
                            x_arrays.append(x[:valid_len])
                            y_arrays.append(y[:valid_len])
                    if not x_arrays or not y_arrays:
                        continue

                    # Use shared edges across the two radius definitions.
                    xvals = np.concatenate(x_arrays)
                    yvals = np.concatenate(y_arrays)
                    if metric == "iso_fraction":
                        yvals = yvals[yvals >= 0]
                    if not len(yvals):
                        continue
                    xlo, xhi = np.min(xvals), np.max(xvals)
                    ylo, yhi = np.min(yvals), np.max(yvals)
                    if xhi <= xlo: xhi = xlo + 1.0
                    if yhi <= ylo: yhi = ylo + 1.0
                    xedges = np.linspace(xlo, xhi, 41)
                    if metric == "iso_fraction":
                        yedges = np.linspace(max(0, ylo), min(1.0, yhi) if yhi > 0 else 1.0, 41)
                    else:
                        yedges = np.linspace(ylo, yhi, 41)

                    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
                    max_count = 0
                    histograms = []
                    for ax, mode in zip(axes, RADIUS_MODES):
                        x = np.asarray(modes[mode][f"{tag}_{xkey}"], dtype=float)
                        y = np.asarray(modes[mode][f"{tag}_{metric}"], dtype=float)
                        valid = np.isfinite(x) & np.isfinite(y)
                        if metric == "iso_fraction":
                            valid &= y >= 0
                        if np.any(valid):
                            h = ax.hist2d(x[valid], y[valid], bins=[xedges, yedges],
                                          cmap="viridis", cmin=1)
                            max_count = max(max_count, float(np.max(h[0])))
                            histograms.append(h[3])
                            ax.set_title(f"{mode.capitalize()} radius (N={int(valid.sum())})")
                        else:
                            ax.text(0.5, 0.5, "No finite data", ha="center", va="center",
                                    transform=ax.transAxes)
                            ax.set_title(f"{mode.capitalize()} radius (N=0)")
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(METRICS[metric])
                    # Apply a shared color scale so density differences are visible fairly.
                    for mesh in histograms:
                        mesh.set_clim(1, max(1, max_count))
                    if metric == "iso_fraction":
                        for ax in axes:
                            ax.set_ylim(yedges[0], yedges[-1])
                    fig.suptitle(f"{name} {tag}: {METRICS[metric]} vs {xlabel}")
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(outdir, f"compare_2d_{name}_{tag}_{metric}_vs_{xkey}.png"),
                        dpi=180,
                    )
                    plt.close(fig)


def plot_particle_summary(all_results, metric, outdir, counts=False):
    """Pion/electron and matched/unmatched distributions for each radius mode."""
    for mode in RADIUS_MODES:
        bins = common_bins([
            all_results[pdg][mode][f"{tag}_{metric}"]
            for pdg in all_results for tag in TAGS
        ])
        if bins is None:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for pdg, modes in all_results.items():
            name, color = particles[pdg]["name"], particles[pdg]["color"]
            for tag in TAGS:
                vals = finite(modes[mode][f"{tag}_{metric}"])
                if metric == "iso_fraction":
                    vals = vals[vals >= 0]
                if not len(vals):
                    continue
                ax.hist(vals, bins=bins, histtype="step",
                        linewidth=2 if tag == "matched" else 1.5,
                        color=color, linestyle="-" if tag == "matched" else "--",
                        density=not counts,
                        label=f"{name} {tag} (N={len(vals)})")
        ax.set_xlabel(METRICS[metric])
        ax.set_ylabel("Counts" if counts else "Normalized density")
        ax.set_title(f"{METRICS[metric]}: matched vs unmatched ({mode} radius)")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{mode}_{metric}_matched_unmatched.png"), dpi=170)
        plt.close(fig)


def plot_original_style_2d(res, info, mode, outdir):
    """Per-particle 2D views similar to the original script, for both radius modes."""
    for metric in METRICS:
        for tag in TAGS:
            for xkey, (xlabel, _) in X_VARIABLES.items():
                x = np.asarray(res[f"{tag}_{xkey}"], dtype=float)
                y = np.asarray(res[f"{tag}_{metric}"], dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if metric == "iso_fraction":
                    valid &= y >= 0
                if not np.any(valid):
                    continue
                fig, ax = plt.subplots(figsize=(6.5, 5))
                hist = ax.hist2d(x[valid], y[valid], bins=40, cmap="viridis",
                                 norm=LogNorm(), cmin=1)
                fig.colorbar(hist[3], ax=ax, label="clusters (log scale)")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(METRICS[metric])
                ax.set_title(f"{info['name']} {tag} ({mode} radius): {METRICS[metric]}\nN={int(valid.sum())}")
                fig.tight_layout()
                fig.savefig(os.path.join(
                    outdir, f"{info['name']}_{mode}_{tag}_{metric}_vs_{xkey}.png"
                ), dpi=170)
                plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# -------------------------------------------------------s--------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="isolation_comparison_2",
                        help="directory for output PNGs")
    parser.add_argument("--max-events", type=int, default=None,
                        help="optionally limit processing to the first N events")
    parser.add_argument("--counts", action="store_true",
                        help="plot raw counts rather than normalized 1D distributions")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_results = {}
    for pdg, info in particles.items():
        print(f"\nProcessing {info['name']} from {info['file']}")
        if not os.path.exists(info["file"]):
            raise FileNotFoundError(
                f"Input ROOT file not found: {info['file']}\n"
                "Edit the file path in the particles configuration at the top of the script."
            )
        all_results[pdg] = process_particle(info, args.max_events)
        moment_difference(all_results[pdg], args.outdir, info["name"])
        for mode in RADIUS_MODES:
            n_matched = len(finite(all_results[pdg][mode]["matched_iso_all"]))
            n_unmatched = len(finite(all_results[pdg][mode]["unmatched_iso_all"]))
            print(f"  {info['name']} {mode}: {n_matched} matched and "
                  f"{n_unmatched} unmatched cluster entries")
            plot_original_style_2d(all_results[pdg][mode], info, mode, args.outdir)

    for metric in METRICS:
        plot_1d_comparison(all_results, metric, args.outdir, counts=args.counts)
        plot_particle_summary(all_results, metric, args.outdir, counts=args.counts)
    #plot_2d_comparisons(all_results, args.outdir)
    plot_radius_vs_isofrac(all_results, args.outdir)
    plot_added_energy_fraction(all_results, args.outdir)
    print(f"\nDone. Comparison plots saved in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
