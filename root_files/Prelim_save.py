#!/usr/bin/env python
"""
Collect calibPrompt energy for the max / min iso_all cluster of each event
(or the single cluster, for one-cluster events).

Per event (every event with >= 1 Pandora cluster that has non-muon hits):
  * 1 cluster            -> one record, tag SINGLE
  * 2+ clusters          -> clusters ranked by iso_all; one record for the max,
                            one for the min (tag MAX / MIN). If the max and min
                            are the same cluster (only one cluster has a valid
                            iso_all) a single record is written, tag BOTH.

calibPrompt of a cluster = sum of the ...CalibPrompt energy over the cells hit by
that cluster (matched by cellID), exactly as match_energy_fast did.

Output (one .npz per input file, plus one combined .npz per particle type):
  pdg, file_idx, event, cluster, tag, n_clusters,
  calib_prompt, iso_all, event_prompt_sum, mc_energy, theta_deg

Usage:
  python collect_prompt_iso.py                 # everything, then combine
  python collect_prompt_iso.py --task 3        # only file #3 of the job list (Slurm array)
  python collect_prompt_iso.py --combine       # just merge existing per-file outputs
  python collect_prompt_iso.py --list          # print the job list
"""
import argparse
import os
import sys
import time

import awkward as ak
import numpy as np
import uproot

# =========================
# CONFIG
# =========================
BASE = "/users/rldohert/data/mucoll/rldohert"
BATCH = 200                   # events read from disk at a time

# Isolation radius floor, same as the MLP script: max_R is switched on only for
# clusters with sigma_T <= SIGMA_T_CUT, and then the radius is floored at MIN_RADIUS.
MIN_RADIUS = 6.0
SIGMA_T_CUT = 12.0
OUT_DIR = "prompt_records"

TAG_SINGLE, TAG_MAX, TAG_MIN, TAG_BOTH = 0, 1, 2, 3
TAG_NAMES = {0: "single", 1: "max", 2: "min", 3: "max_and_min"}

SYSTEM2NAME = {
    679272617:  "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
REAL_SYSTEMS = [
    "EcalBarrelCollectionRec", "HcalBarrelCollectionRec",
    "EcalEndcapCollectionRec", "HcalEndcapCollectionRec",
]
# reco system -> its CalibPrompt collection
SYS_TO_CALIB = {
    "EcalBarrelCollectionRec": "ECalBarrelCollectionCalibPrompt",
    "HcalBarrelCollectionRec": "HCalBarrelCollectionCalibPrompt",
    "EcalEndcapCollectionRec": "ECalEndcapCollectionCalibPrompt",
    "HcalEndcapCollectionRec": "HCalEndcapCollectionCalibPrompt",
}

_TAIL = "pt_10_200_theta_45-135"
SAMPLES = {
    11: [
        f"{BASE}/pdg_11_{_TAIL}_calib/reco_pdg_11_{_TAIL}_nobib.root",
        f"{BASE}/pdg_11_{_TAIL}_calib2/reco_pdg_11_{_TAIL}_nobib.root",
    ],
    211: [
        f"{BASE}/pdg_211_{_TAIL}_{calib}/reco_pdg_211_{_TAIL}_nobib_part{p}.root"
        for calib in ("calib2", "calib3") for p in (1, 2, 3)
    ],
}
JOBS = [(pdg, i, path) for pdg, paths in SAMPLES.items() for i, path in enumerate(paths)]

# =========================
# BRANCHES
# =========================
def build_branches():
    br = {
        "hits_begin": "PandoraClusters/PandoraClusters.hits_begin",
        "hits_end":   "PandoraClusters/PandoraClusters.hits_end",
        "hit_index":  "_PandoraClusters_hits/_PandoraClusters_hits.index",
        "hit_coll":   "_PandoraClusters_hits/_PandoraClusters_hits.collectionID",
        "status":     "MCParticles/MCParticles.generatorStatus",
        "px":         "MCParticles/MCParticles.momentum.x",
        "py":         "MCParticles/MCParticles.momentum.y",
        "pz":         "MCParticles/MCParticles.momentum.z",
        "mass":       "MCParticles/MCParticles.mass",
    }
    for s in REAL_SYSTEMS:
        for key, field in (("ene", "energy"), ("cid", "cellID"),
                           ("x", "position.x"), ("y", "position.y"), ("z", "position.z")):
            br[f"{key}|{s}"] = f"{s}/{s}.{field}"
    for s, c in SYS_TO_CALIB.items():
        br[f"cene|{s}"] = f"{c}/{c}.energy"
        br[f"ccid|{s}"] = f"{c}/{c}.cellID"
    return br

BRANCHES = build_branches()

# =========================
# HELPERS
# =========================
def to_np(a, dtype=None):
    arr = np.asarray(ak.to_numpy(a))
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def build_layer_groups(layer):
    """{layer_number: array of hit indices in that layer}"""
    if len(layer) == 0:
        return {}
    order = np.argsort(layer, kind="stable")
    sl = layer[order]
    cuts = np.nonzero(np.diff(sl))[0] + 1
    groups = np.split(order, cuts)
    keys = sl[np.concatenate(([0], cuts))]
    return {int(k): g for k, g in zip(keys, groups)}


def event_arrays(b, i):
    """Pull the hit arrays of event i into plain numpy, once per event."""
    ev = {"sys": {}, "calib": {}}
    for s in REAL_SYSTEMS:
        cid = to_np(b[f"cid|{s}"][i]).astype(np.uint64)
        layer = ((cid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
        ev["sys"][s] = dict(
            e=to_np(b[f"ene|{s}"][i], float),
            x=to_np(b[f"x|{s}"][i], float),
            y=to_np(b[f"y|{s}"][i], float),
            z=to_np(b[f"z|{s}"][i], float),
            cid=cid,
            layer=layer,
            groups=build_layer_groups(layer),
        )
        ev["calib"][s] = dict(
            e=to_np(b[f"cene|{s}"][i], float),
            cid=to_np(b[f"ccid|{s}"][i]).astype(np.uint64),
        )
    return ev


def cluster_hits(c, hb, he, hidx, hcoll):
    """{system: unique hit indices} for cluster c, muon-system hits dropped."""
    s, e = int(hb[c]), int(he[c])
    if e <= s:
        return None
    idx = hidx[s:e]
    ids = hcoll[s:e]
    own = {}
    for sid in np.unique(ids):
        name = SYSTEM2NAME.get(int(sid))
        if name is None or name == "Skip":
            continue
        own[name] = np.unique(idx[ids == sid])
    return own or None


def cluster_prompt(own, ev):
    """Sum of CalibPrompt energy over the cells the cluster touches (matched by cellID)."""
    total = 0.0
    for s, own_idx in own.items():
        calib = ev["calib"][s]
        if calib["cid"].size == 0:
            continue
        cids = np.unique(ev["sys"][s]["cid"][own_idx])
        total += float(calib["e"][np.isin(calib["cid"], cids)].sum())
    return total


def cluster_sigma_T(own, ev):
    """sqrt(eig0 + eig1) of the energy-weighted inertia tensor (same as get_shower_axis)."""
    xs, ys, zs, ws = [], [], [], []
    for s, own_idx in own.items():
        d = ev["sys"][s]
        xs.append(d["x"][own_idx]); ys.append(d["y"][own_idx])
        zs.append(d["z"][own_idx]); ws.append(d["e"][own_idx])
    xs, ys, zs, ws = (np.concatenate(a) for a in (xs, ys, zs, ws))
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & np.isfinite(ws)
    xs, ys, zs, ws = xs[ok], ys[ok], zs[ok], ws[ok]
    W = ws.sum()
    if ws.size == 0 or not np.isfinite(W) or W <= 0:
        return np.nan
    pts = np.vstack([xs, ys, zs]).T
    cog = (pts * ws[:, None]).sum(axis=0) / W
    diffs = pts - cog
    inertia = np.einsum("ni,nj,n->ij", diffs, diffs, ws) / W
    ev_vals = np.linalg.eigvalsh(inertia)
    return float(np.sqrt(max(ev_vals[0] + ev_vals[1], 0.0)))


def use_max_R(own, ev):
    """max_R = (sigma_T <= SIGMA_T_CUT), as in the MLP script (False if sigma_T is not finite)."""
    sig = cluster_sigma_T(own, ev)
    return bool(np.isfinite(sig) and sig <= SIGMA_T_CUT)


def cluster_iso_all(own, ev, max_R):
    """
    E-weighted mean number of non-cluster hits within `radius` of the cluster's hits,
    per (system, layer) the cluster touches. Same logic as isolation_allhits_but
    (iso_all only, so no 'unclaimed' bookkeeping is needed).
    """
    num = den = 0.0
    for s, own_idx in own.items():
        d = ev["sys"][s]
        e, px, py, pz, layer = d["e"], d["x"], d["y"], d["z"], d["layer"]
        n = len(e)
        own_idx = own_idx[(own_idx >= 0) & (own_idx < n)]
        if own_idx.size == 0:
            continue
        in_own = np.zeros(n, dtype=bool)
        in_own[own_idx] = True

        for lay in np.unique(layer[own_idx]):
            grp = d["groups"][int(lay)]
            mine = grp[in_own[grp]]
            cand = grp[~in_own[grp]]
            if cand.size == 0 or mine.size == 0:
                continue

            w = e[mine]
            E_s = w.sum()
            if not np.isfinite(E_s) or E_s <= 0:
                continue
            ox, oy, oz = px[mine], py[mine], pz[mine]
            cx, cy, cz = np.sum(w * ox) / E_s, np.sum(w * oy) / E_s, np.sum(w * oz) / E_s
            width = np.sqrt(np.sum(w * ((ox - cx) ** 2 + (oy - cy) ** 2 + (oz - cz) ** 2)) / E_s)
            radius = width / 2
            if max_R and radius < MIN_RADIUS:
                radius = MIN_RADIUS

            nx, ny, nz = px[cand], py[cand], pz[cand]
            d2 = ((ox[:, None] - nx[None, :]) ** 2
                  + (oy[:, None] - ny[None, :]) ** 2
                  + (oz[:, None] - nz[None, :]) ** 2)
            n_neighbor = int((d2 < radius ** 2).any(axis=0).sum())

            num += E_s * n_neighbor
            den += E_s
    return num / den if den > 0 else np.nan


def mc_truth(b, i):
    """(true energy summed over final-state MC particles, polar angle [deg] of the first one)."""
    mask = to_np(b["status"][i]) == 1
    if not mask.any():
        return np.nan, np.nan
    px = to_np(b["px"][i], float)[mask]
    py = to_np(b["py"][i], float)[mask]
    pz = to_np(b["pz"][i], float)[mask]
    m = to_np(b["mass"][i], float)[mask]
    e = float(np.sum(np.sqrt(m ** 2 + px ** 2 + py ** 2 + pz ** 2)))
    p0 = np.sqrt(px[0] ** 2 + py[0] ** 2 + pz[0] ** 2)
    theta = float(np.degrees(np.arccos(pz[0] / p0))) if p0 > 0 else np.nan
    return (e if e > 0 else np.nan), theta


def process_event(b, i):
    """Returns (list of (cluster, tag, prompt, iso_all), n_clusters, event_prompt_sum) or (None, reason)."""
    hb = to_np(b["hits_begin"][i])
    he = to_np(b["hits_end"][i])
    n_cl = len(hb)
    if n_cl == 0:
        return None, "no_clusters"

    hidx = to_np(b["hit_index"][i]).astype(np.int64)
    hcoll = to_np(b["hit_coll"][i])
    ev = event_arrays(b, i)

    clus = {}                                   # cluster number -> (prompt, iso_all)
    for c in range(n_cl):
        own = cluster_hits(c, hb, he, hidx, hcoll)
        if own is None:
            continue
        prompt = cluster_prompt(own, ev)
        iso = cluster_iso_all(own, ev, use_max_R(own, ev))
        clus[c] = (prompt, iso)
    if not clus:
        return None, "no_usable_clusters"

    event_sum = float(sum(p for p, _ in clus.values()))

    if n_cl == 1:
        picks = [(0, TAG_SINGLE)]
    else:
        valid = [c for c, (_, iso) in clus.items() if np.isfinite(iso)]
        if not valid:
            return None, "no_valid_iso"
        c_max = max(valid, key=lambda c: clus[c][1])
        c_min = min(valid, key=lambda c: clus[c][1])
        picks = [(c_max, TAG_BOTH)] if c_max == c_min else [(c_max, TAG_MAX), (c_min, TAG_MIN)]

    return ([(c, tag, clus[c][0], clus[c][1]) for c, tag in picks], n_cl, event_sum), None


# =========================
# FILE LOOP
# =========================
COLS = ["pdg", "file_idx", "event", "cluster", "tag", "n_clusters",
        "calib_prompt", "iso_all", "event_prompt_sum", "mc_energy", "theta_deg"]
DTYPES = dict(pdg=np.int32, file_idx=np.int16, event=np.int64, cluster=np.int32,
              tag=np.int8, n_clusters=np.int32, calib_prompt=np.float64, iso_all=np.float64,
              event_prompt_sum=np.float64, mc_energy=np.float64, theta_deg=np.float64)


def out_path(out_dir, pdg, path):
    tag = os.path.basename(os.path.dirname(path)) + "__" + os.path.splitext(os.path.basename(path))[0]
    return os.path.join(out_dir, f"{tag}.npz")


def process_file(pdg, file_idx, path, batch=BATCH):
    rows = {k: [] for k in COLS}
    skipped = {}
    t0 = time.time()
    with uproot.open(path) as f:
        tree = f["events"]
        n_ev = tree.num_entries
        print(f"  {n_ev:,} entries", flush=True)
        for start in range(0, n_ev, batch):
            stop = min(start + batch, n_ev)
            b = {k: tree[p].array(entry_start=start, entry_stop=stop) for k, p in BRANCHES.items()}
            for j in range(stop - start):
                res, why = process_event(b, j)
                if res is None:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                picks, n_cl, event_sum = res
                mc_e, th = mc_truth(b, j)
                for c, tag, prompt, iso in picks:
                    rows["pdg"].append(pdg); rows["file_idx"].append(file_idx)
                    rows["event"].append(start + j); rows["cluster"].append(c)
                    rows["tag"].append(tag); rows["n_clusters"].append(n_cl)
                    rows["calib_prompt"].append(prompt); rows["iso_all"].append(iso)
                    rows["event_prompt_sum"].append(event_sum)
                    rows["mc_energy"].append(mc_e); rows["theta_deg"].append(th)
            done = stop
            print(f"  {done:,}/{n_ev:,} events | {len(rows['event']):,} records | "
                  f"{time.time() - t0:.0f}s", flush=True)
    arrays = {k: np.asarray(v, dtype=DTYPES[k]) for k, v in rows.items()}
    return arrays, skipped


def save_npz(path, arrays, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays, **extra)
    os.replace(tmp, path)


def combine(out_dir):
    for pdg, paths in SAMPLES.items():
        parts = [out_path(out_dir, pdg, p) for p in paths]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            print(f"[combine] pdg {pdg}: {len(missing)} of {len(parts)} per-file outputs missing, skipping")
            continue
        merged = {k: [] for k in COLS}
        for p in parts:
            with np.load(p) as d:
                for k in COLS:
                    merged[k].append(d[k])
        arrays = {k: np.concatenate(v) for k, v in merged.items()}
        dest = os.path.join(out_dir, f"prompt_records_pdg{pdg}.npz")
        save_npz(dest, arrays, files=np.array(paths), tag_names=np.array(
            [f"{k}={v}" for k, v in TAG_NAMES.items()]))
        n = len(arrays["event"])
        counts = {TAG_NAMES[t]: int((arrays["tag"] == t).sum()) for t in TAG_NAMES}
        print(f"[combine] pdg {pdg}: {n:,} records -> {dest}  {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, help="process only file #N of the job list")
    ap.add_argument("--combine", action="store_true", help="only merge existing per-file outputs")
    ap.add_argument("--list", action="store_true", help="print the job list and exit")
    ap.add_argument("--force", action="store_true", help="redo files that already have output")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    out_dir = OUT_DIR

    if args.list:
        for n, (pdg, i, path) in enumerate(JOBS):
            print(f"{n}\tpdg {pdg}\t{path}")
        return
    if args.combine:
        combine(out_dir)
        return

    jobs = list(enumerate(JOBS))
    if args.task is not None:
        if not 0 <= args.task < len(JOBS):
            sys.exit(f"--task must be 0..{len(JOBS) - 1}")
        jobs = [jobs[args.task]]

    print(f"max_R: on when sigma_T <= {SIGMA_T_CUT} | output dir: {out_dir}")
    for n, (pdg, file_idx, path) in jobs:
        dest = out_path(out_dir, pdg, path)
        print(f"\n[{n}] pdg {pdg} | {path}")
        if os.path.exists(dest) and not args.force:
            print(f"  already done -> {dest}")
            continue
        if not os.path.exists(path):
            print("  INPUT NOT FOUND, skipping")
            continue
        arrays, skipped = process_file(pdg, file_idx, path, args.batch)
        save_npz(dest, arrays)
        print(f"  saved {len(arrays['event']):,} records -> {dest}")
        if skipped:
            print(f"  events skipped: {skipped}")

    if args.task is None:
        combine(out_dir)


if __name__ == "__main__":
    main()