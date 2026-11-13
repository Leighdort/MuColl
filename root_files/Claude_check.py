#!/usr/bin/env python
"""
Diagnostics: Pandora clusters -> Rec hits -> sim hits (-> MCParticles)

For each event / cluster it reports:
  * cluster energy vs. sum of its Rec-hit energies vs. sum of its (unique) sim-hit energies
  * how many sim hits each Rec hit maps to, and how many Rec hits have no relation
  * how many sim hits are shared between clusters (truth split across clusters)
  * how much sim energy in the event is NOT in any cluster
  * which MCParticles deposited the cluster's sim energy (via the Contributions collections)
  * which PFO(s) the cluster belongs to
"""
import csv
from collections import defaultdict, Counter

import numpy as np
import awkward as ak
import uproot

# ----------------------------------------------------------------- config
FILE = ("/users/rldohert/data/mucoll/rldohert/"
        "pdg_211_pt_50_theta_85-85_testing_fede2/job_0/"
        "reco_output_p50_211_nobib0.edm4hep.root")
EVENTS = list(range(940, 950))          # 10 events
OUT_CSV = "cluster_sim_diagnostics.csv"

# collectionID -> Rec collection name (these IDs can differ between files!)
system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec",
                "EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]

CL = "PandoraClusters/PandoraClusters"
CLH = "_PandoraClusters_hits/_PandoraClusters_hits"
PFO = "PandoraPFOs/PandoraPFOs"
PFOC = "_PandoraPFOs_clusters/_PandoraPFOs_clusters"
MC = "MCParticles/MCParticles"


# ----------------------------------------------------------------- reader
class Reader:
    """Loads each branch once for the entry range, looks keys up case-insensitively."""

    def __init__(self, tree, start, stop):
        self.tree, self.start, self.stop = tree, start, stop
        self.keys = {k.lower(): k for k in tree.keys()}
        self.cache = {}

    def has(self, key):
        return key.lower() in self.keys

    def get(self, key, k):
        """Branch `key`, entry number k (relative to self.start), as a numpy array."""
        real = self.keys[key.lower()]
        if real not in self.cache:
            self.cache[real] = self.tree[real].array(
                entry_start=self.start, entry_stop=self.stop)
        return ak.to_numpy(self.cache[real][k])


# ----------------------------------------------------------------- per-system
def build_system(R, k, rec_name):
    """Collect Rec hits, SimRec relations, sim hits (+ contributions) for one calorimeter."""
    base = rec_name[:-3]                                         # EcalBarrelCollection
    rel = rec_name.replace("CollectionRec", "RelationsSimRec")   # EcalBarrelRelationsSimRec
    need = {
        "rec_E": f"{rec_name}/{rec_name}.energy",
        "sim_E": f"{base}/{base}.energy",
        "w":     f"{rel}/{rel}.weight",
        "from":  f"_{rel}_from/_{rel}_from.index",
        "to":    f"_{rel}_to/_{rel}_to.index",
    }
    missing = [v for v in need.values() if not R.has(v)]
    if missing:
        return None, missing

    s = {"rec_E": R.get(need["rec_E"], k), "sim_E": R.get(need["sim_E"], k)}
    w = R.get(need["w"], k)
    frm = R.get(need["from"], k).astype(int)
    to = R.get(need["to"], k).astype(int)

    s["rec2sim"] = defaultdict(list)
    for f, t, ww in zip(frm, to, w):
        s["rec2sim"][int(f)].append((int(t), float(ww)))
    s["sim_nrec"] = Counter(to.tolist())        # how many Rec hits each sim hit feeds

    for tag in ("from", "to"):                  # sanity: which collection IDs show up
        key = f"_{rel}_{tag}/_{rel}_{tag}.collectionID"
        s[f"{tag}_cid"] = sorted(set(R.get(key, k).tolist())) if R.has(key) else []

    ck = {
        "begin": f"{base}/{base}.contributions_begin",
        "end":   f"{base}/{base}.contributions_end",
        "cidx":  f"_{base}_contributions/_{base}_contributions.index",
        "cen":   f"{base}Contributions/{base}Contributions.energy",
        "cpart": f"_{base}Contributions_particle/_{base}Contributions_particle.index",
    }
    if all(R.has(v) for v in ck.values()):
        c = {n: R.get(v, k) for n, v in ck.items()}
        for n in ("begin", "end", "cidx", "cpart"):
            c[n] = c[n].astype(int)
        s["contrib"] = c
    return s, []


def event_mc(R, k):
    keys = [f"{MC}.PDG", f"{MC}.momentum.x", f"{MC}.momentum.y",
            f"{MC}.momentum.z", f"{MC}.mass", f"{MC}.generatorStatus"]
    if not all(R.has(x) for x in keys):
        return None
    pdg = R.get(keys[0], k)
    p = [R.get(keys[i], k) for i in (1, 2, 3)]
    m = R.get(keys[4], k)
    return {"pdg": pdg, "E": np.sqrt(p[0]**2 + p[1]**2 + p[2]**2 + m**2),
            "status": R.get(keys[5], k)}


# ----------------------------------------------------------------- per-event
def analyse_event(R, k, evt, rows):
    print("\n" + "=" * 100)
    print(f"EVENT {evt}")
    print("=" * 100)

    cl_E = R.get(f"{CL}.energy", k)
    hb = R.get(f"{CL}.hits_begin", k).astype(int)
    he = R.get(f"{CL}.hits_end", k).astype(int)
    hidx = R.get(f"{CLH}.index", k).astype(int)
    hcid = R.get(f"{CLH}.collectionID", k)
    ncl = len(cl_E)

    # cluster -> PFO(s)
    cl2pfo = defaultdict(list)
    if R.has(f"{PFO}.clusters_begin"):
        pb = R.get(f"{PFO}.clusters_begin", k).astype(int)
        pe = R.get(f"{PFO}.clusters_end", k).astype(int)
        pci = R.get(f"{PFOC}.index", k).astype(int)
        pE = R.get(f"{PFO}.energy", k)
        pPDG = R.get(f"{PFO}.PDG", k)
        for ip in range(len(pb)):
            for c in pci[pb[ip]:pe[ip]]:
                cl2pfo[int(c)].append((int(pPDG[ip]), float(pE[ip])))

    mc = event_mc(R, k)
    if mc is not None:
        gun = mc["status"] == 1
        # only count particles without a daughter-of-gun ambiguity: just list gen-status-1 ones
        print(f"MC gen-status-1 particles: n={gun.sum()}, "
              f"sum E = {mc['E'][gun].sum():.3f} GeV, "
              f"PDGs = {sorted(set(mc['pdg'][gun].tolist()))}")
    print(f"N clusters = {ncl}")

    # build per-calorimeter info
    systems = {}
    for name in real_systems:
        s, missing = build_system(R, k, name)
        if s is None:
            print(f"  [warn] {name}: skipping, missing branches {missing}")
            continue
        systems[name] = s
        print(f"  {name}: nRec={len(s['rec_E'])} nSim={len(s['sim_E'])} "
              f"relFromCID={s['from_cid']} relToCID={s['to_cid']} "
              f"contributions={'yes' if 'contrib' in s else 'NO'}")

    sim2clusters = defaultdict(set)      # (system, sim idx) -> clusters using it
    rec2clusters = defaultdict(set)      # (system, rec idx) -> clusters using it
    per_cluster = []

    for c in range(ncl):
        cids = hcid[hb[c]:he[c]]
        idxs = hidx[hb[c]:he[c]]
        r = dict(event=evt, cluster=c, E_cluster=float(cl_E[c]), n_hits=len(idxs),
                 n_skipped=0, n_unknown=0, E_rec_hits=0.0, E_sim=0.0, E_sim_weighted=0.0,
                 n_norel=0, n_sim_unique=0, max_sim_per_rec=0)
        sim_keys = set()
        n_sim_per_rec = []
        pE = defaultdict(float)
        nsys = Counter()

        for cid, i in zip(cids, idxs):
            name = system2name.get(int(cid))
            if name == "Skip":
                r["n_skipped"] += 1
                continue
            if name is None or name not in systems:
                r["n_unknown"] += 1
                continue
            s = systems[name]
            nsys[name] += 1
            r["E_rec_hits"] += float(s["rec_E"][i])
            rec2clusters[(name, int(i))].add(c)
            rels = s["rec2sim"].get(int(i), [])
            if not rels:
                r["n_norel"] += 1
            n_sim_per_rec.append(len(rels))
            for t, w in rels:
                sim_keys.add((name, t))
                r["E_sim_weighted"] += w * float(s["sim_E"][t])

        for (name, t) in sim_keys:
            s = systems[name]
            r["E_sim"] += float(s["sim_E"][t])
            sim2clusters[(name, t)].add(c)
            if "contrib" in s:
                cc = s["contrib"]
                for j in range(cc["begin"][t], cc["end"][t]):
                    ci = cc["cidx"][j]
                    pE[int(cc["cpart"][ci])] += float(cc["cen"][ci])

        r["n_sim_unique"] = len(sim_keys)
        r["max_sim_per_rec"] = max(n_sim_per_rec) if n_sim_per_rec else 0
        r["hits_by_system"] = dict(nsys)
        r["pE"] = pE
        r["sim_keys"] = sim_keys
        per_cluster.append(r)

    # second pass: sharing between clusters + printing
    print(f"\n{'cl':>3} {'E_cl':>8} {'E_recHits':>9} {'E_sim':>8} {'sim/cl':>7} {'nHit':>5} "
          f"{'nSim':>5} {'noRel':>5} {'maxS/R':>6} {'shared%':>8}  PFO(pdg,E)")
    for r in per_cluster:
        shared_E = 0.0
        n_shared = 0
        for key in r["sim_keys"]:
            if len(sim2clusters[key]) > 1:
                n_shared += 1
                shared_E += float(systems[key[0]]["sim_E"][key[1]])
        r["n_sim_shared"] = n_shared
        r["shared_frac"] = shared_E / r["E_sim"] if r["E_sim"] > 0 else 0.0
        ratio = r["E_sim"] / r["E_cluster"] if r["E_cluster"] > 0 else float("nan")
        pf = ", ".join(f"({p},{e:.1f})" for p, e in cl2pfo.get(r["cluster"], [])) or "-"
        print(f"{r['cluster']:>3} {r['E_cluster']:8.3f} {r['E_rec_hits']:9.3f} {r['E_sim']:8.4f} "
              f"{ratio:7.4f} {r['n_hits']:5d} {r['n_sim_unique']:5d} {r['n_norel']:5d} "
              f"{r['max_sim_per_rec']:6d} {100*r['shared_frac']:7.1f}%  {pf}")
        extra = []
        if r["n_skipped"]:
            extra.append(f"skipped={r['n_skipped']}")
        if r["n_unknown"]:
            extra.append(f"UNKNOWN collID hits={r['n_unknown']}")
        extra.append("hits/system=" + str(r["hits_by_system"]))
        print("      " + "  ".join(extra))

        # truth particles
        if r["pE"]:
            tot = sum(r["pE"].values())
            top = sorted(r["pE"].items(), key=lambda kv: -kv[1])[:3]
            parts = []
            for idx, e in top:
                pdg = int(mc["pdg"][idx]) if mc is not None and idx < len(mc["pdg"]) else "?"
                mcE = f"{mc['E'][idx]:.1f}" if mc is not None and idx < len(mc["E"]) else "?"
                parts.append(f"MC#{idx} pdg={pdg} {e:.4f}GeV ({100*e/tot:.0f}%, trueE={mcE})")
            print("      truth: " + " | ".join(parts))
            r["top_mc_idx"] = top[0][0]
            r["top_mc_frac"] = top[0][1] / tot
            r["n_mc_particles"] = len(r["pE"])
        else:
            r["top_mc_idx"], r["top_mc_frac"], r["n_mc_particles"] = -1, float("nan"), 0

    # event summary
    print("\n  -- event-level checks --")
    for name, s in systems.items():
        used = {t for r in per_cluster for (n, t) in r["sim_keys"] if n == name}
        tot = float(s["sim_E"].sum())
        clu = float(sum(s["sim_E"][t] for t in used))
        multi = sum(1 for v in s["sim_nrec"].values() if v > 1)
        rec_shared = sum(1 for (n, _), cs in rec2clusters.items() if n == name and len(cs) > 1)
        frac = 100 * clu / tot if tot > 0 else 0.0
        print(f"  {name[:-3]:<22} total sim E={tot:9.4f}  in clusters={clu:9.4f} ({frac:5.1f}%)  "
              f"sim hits->>1 Rec hit: {multi}  Rec hits in >1 cluster: {rec_shared}")
    n_shared_sim = sum(1 for cs in sim2clusters.values() if len(cs) > 1)
    print(f"  sim hits used by >1 cluster: {n_shared_sim} of {len(sim2clusters)}")
    tot_cl = sum(r["E_cluster"] for r in per_cluster)
    tot_sim = sum(r["E_sim"] for r in per_cluster if True)
    print(f"  sum cluster E = {tot_cl:.3f} GeV; sum cluster sim E = {tot_sim:.4f} GeV "
          f"(may double count shared hits)")

    for r in per_cluster:
        rows.append({k2: v for k2, v in r.items()
                     if k2 not in ("pE", "sim_keys", "hits_by_system")})


# ----------------------------------------------------------------- main
def main():
    f = uproot.open(FILE)
    tree = f["events"]
    start, stop = min(EVENTS), max(EVENTS) + 1
    R = Reader(tree, start, stop)

    # quick presence check
    for key in (f"{CL}.energy", f"{CL}.hits_begin", f"{CLH}.index", f"{CLH}.collectionID"):
        if not R.has(key):
            raise SystemExit(f"Missing branch: {key}")

    rows = []
    for evt in EVENTS:
        analyse_event(R, evt - start, evt, rows)

    if rows:
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} cluster rows to {OUT_CSV}")


if __name__ == "__main__":
    main()