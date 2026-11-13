#!/usr/bin/env python
"""
Standalone checks on the merged reco files:
  1. entry counts
  2. PFO multiplicity (0 / 1 / 2+) and what the extra PFOs are
  3. whether two files contain the SAME events (duplicate seeds)

Usage:
    python check_files.py                 # uses the default file list below
    python check_files.py fileA.root fileB.root ...
"""
import sys
import numpy as np
import uproot
import awkward as ak

base_path = "/users/rldohert/data/mucoll/rldohert"

def merged(pdg, suffix):
    d = f"pdg_{pdg}_pt_10_200_theta_45-135_fede_calib_big{suffix}"
    return f"{base_path}/{d}/reco_pdg_{pdg}_pt_10_200_theta_45-135_nobib.root"

DEFAULT_FILES = [merged(211, ""), merged(211, "_2")]
files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FILES

N_COMPARE = 1000   # events used for the duplicate comparison


def truth_summary(events, n):
    """Per-event truth: summed final-state px, py, pz (identifies an event)."""
    stop = min(n, events.num_entries)
    status = events["MCParticles/MCParticles.generatorStatus"].array(entry_stop=stop)
    out = []
    for comp in ("x", "y", "z"):
        arr = events[f"MCParticles/MCParticles.momentum.{comp}"].array(entry_stop=stop)
        out.append(ak.to_numpy(ak.sum(arr[status == 1], axis=1)))
    return np.column_stack(out)


print("=" * 70)
print("1) ENTRY COUNTS AND PFO MULTIPLICITY")
print("=" * 70)

truth = {}
for path in files:
    print(f"\n{path}")
    with uproot.open(path) as f:
        ev = f["events"]
        n_ev = ev.num_entries
        pfo_e = ev["PandoraPFOs/PandoraPFOs.energy"].array()
        n_pfo = ak.to_numpy(ak.num(pfo_e))
        print(f"  entries: {n_ev}")
        print(f"  PFOs/event: 0 -> {int((n_pfo == 0).sum())}, "
              f"1 -> {int((n_pfo == 1).sum())}, 2+ -> {int((n_pfo >= 2).sum())}")

        # What are the PFOs in events with 2+?
        pid = ak.flatten(ev["PandoraPFOs/PandoraPFOs.PDG"].array()[n_pfo >= 2])
        pid = ak.to_numpy(pid)
        if len(pid):
            vals, cnts = np.unique(pid, return_counts=True)
            order = np.argsort(-cnts)[:8]
            print("  PDG of PFOs in 2+ PFO events (top 8):")
            for i in order:
                print(f"     PDG {int(vals[i]):>6}: {int(cnts[i])}")

        truth[path] = truth_summary(ev, N_COMPARE)


print("\n" + "=" * 70)
print(f"2) DUPLICATE CHECK (first {N_COMPARE} events, truth momentum)")
print("=" * 70)

paths = list(truth.keys())
for i in range(len(paths)):
    for j in range(i + 1, len(paths)):
        a, b = truth[paths[i]], truth[paths[j]]
        n = min(len(a), len(b))
        same_order = np.isclose(a[:n], b[:n], rtol=0, atol=1e-6).all(axis=1).mean()
        # also check for the same events in a different order
        keys_a = {tuple(np.round(r, 5)) for r in a}
        overlap = np.mean([tuple(np.round(r, 5)) in keys_a for r in b])
        print(f"\nA: {paths[i]}")
        print(f"B: {paths[j]}")
        print(f"  identical event-by-event (same order): {same_order:.1%}")
        print(f"  B events also present anywhere in A:   {overlap:.1%}")
        if same_order > 0.9 or overlap > 0.9:
            print("  --> DUPLICATE SAMPLES (same seeds). Do not treat as independent.")
        else:
            print("  --> looks independent")