import os
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt

USE_CACHE = False
CACHE_FILE = "mc_spread_cache.npz"   # new name so it doesn't clash with your feature cache

# ################### FILES ##############################
base_path = "/users/rldohert/data/mucoll/rldohert"

def merged(pdg, suffix):
    d = f"pdg_{pdg}_pt_10_200_theta_45-135_fede_calib_big{suffix}"
    return f"{base_path}/{d}/reco_pdg_{pdg}_pt_10_200_theta_45-135_nobib.root"


particles = {
    11:  {"name": "electron", "files": [merged(11, ""),  merged(11, "_2")]},
    211: {"name": "pion",     "files": [merged(211, ""), merged(211, "_2")]},
}
'''
particles = {
    11:  {"name": "electron", "files": [merged(11, ""),  merged(11, "_2")]},
}
'''
particle_types = {11: "Electrons", 211: "Pions"}
colors = {11: "navy", 211: "firebrick"}
KEYS = ["mc_energy", "theta", "e_cluster", "n_clusters"]

# ################### MC TRUTH (vectorised over all events) ##############################
def mc_energy_and_theta(events):
    """Per-event E_true (sum over final-state MC particles) and polar angle [deg]
    of their total momentum. Events with no final-state particle give NaN."""
    mc = events["MCParticles"]
    final = mc["MCParticles.generatorStatus"].array() == 1
    px = mc["MCParticles.momentum.x"].array()
    py = mc["MCParticles.momentum.y"].array()
    pz = mc["MCParticles.momentum.z"].array()
    m  = mc["MCParticles.mass"].array()

    E  = ak.to_numpy(ak.sum(np.sqrt(m**2 + px**2 + py**2 + pz**2)[final], axis=1))
    Px = ak.to_numpy(ak.sum(px[final], axis=1))
    Py = ak.to_numpy(ak.sum(py[final], axis=1))
    Pz = ak.to_numpy(ak.sum(pz[final], axis=1))
    p = np.sqrt(Px**2 + Py**2 + Pz**2)

    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.degrees(np.arccos(np.clip(Pz / p, -1, 1)))
    E = np.where(E > 0, E, np.nan)
    return E, theta

# ################### EXTRACTION ##############################
def load_particle(info):
    rows = {"mc_energy": [], "theta": [], "e_cluster": [], "n_clusters": []}  # one row per event
    for file_path in info["files"]:
        print(f"Processing {info['name']}: {file_path}")
        events = uproot.open(file_path)["events"]
        print(f"  entries in file: {events.num_entries}")

        E_mc, theta = mc_energy_and_theta(events)
        n_pfo = ak.to_numpy(ak.num(events["PandoraPFOs/PandoraPFOs.energy"].array()))
        cl_begin = events["PandoraPFOs/PandoraPFOs.clusters_begin"].array()
        cl_end   = events["PandoraPFOs/PandoraPFOs.clusters_end"].array()
        cl_index = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"].array()
        cluster_energy = events["PandoraClusters"]["PandoraClusters.energy"].array()
        for index in range(events.num_entries):
            rows["mc_energy"].append(E_mc[index])
            rows["theta"].append(theta[index])

    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}
if USE_CACHE and os.path.exists(CACHE_FILE):
    cache = np.load(CACHE_FILE)
    all_particle_data = {pdg: {k: cache[f"{pdg}_{k}"] for k in KEYS} for pdg in particles}
    print(f"Loaded from {CACHE_FILE}")
else:
    all_particle_data = {pdg: load_particle(info) for pdg, info in particles.items()}
    np.savez(CACHE_FILE, **{f"{pdg}_{k}": v
                            for pdg, d in all_particle_data.items() for k, v in d.items()})

for pdg, info in particles.items():
    print(f"{info['name']}: {len(all_particle_data[pdg]['mc_energy'])} clusters total")

# ################### COMBINE ##############################
pdg_all   = np.concatenate([np.full(len(d["mc_energy"]), pdg)
                            for pdg, d in all_particle_data.items()])
E_mc      = np.concatenate([d["mc_energy"] for d in all_particle_data.values()])
Angle_mc  = np.concatenate([d["theta"]     for d in all_particle_data.values()])
good = np.isfinite(E_mc) & np.isfinite(Angle_mc)
print(f"Dropping {(~good).sum()} events with no valid final-state MC particle")
pdg_all, E_mc, Angle_mc= (
    pdg_all[good], E_mc[good], Angle_mc[good])
def make_edges(lo, hi, width):
    """Fixed-width edges starting at lo, extended until they cover hi."""
    n = int(np.ceil((hi - lo) / width - 1e-9))
    return lo + width * np.arange(n + 1)

E_edges_5,  E_edges_20 = make_edges(10, E_mc.max(), 5),  make_edges(10, E_mc.max(), 20)
A_edges_5,  A_edges_10 = make_edges(45, Angle_mc.max(), 5), make_edges(45, Angle_mc.max(), 10)

# ################### PRINT SUMMARY ##############################
print(f"{'Particle':<10}{'N':>8}{'E mean':>9}{'E std':>9}{'E min':>8}{'E max':>8}"
      f"{'Ang mean':>10}{'Ang std':>9}{'Ang min':>9}{'Ang max':>9}")
for pdg, name in particle_types.items():
    m = pdg_all == pdg
    E, A = E_mc[m], Angle_mc[m]
    print(f"{name:<10}{m.sum():>8}{E.mean():>9.2f}{E.std():>9.2f}{E.min():>8.1f}{E.max():>8.1f}"
          f"{A.mean():>10.2f}{A.std():>9.2f}{A.min():>9.1f}{A.max():>9.1f}")

# ################### 1D SPREADS ##############################
def spread_hist(values_by_pdg, bins, xlabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    for pdg, name in particle_types.items():
        v = values_by_pdg[pdg]
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.45, color=colors[pdg],
                label=f"{name} (N={len(v)}, mean={v.mean():.2f}, σ={v.std():.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events per Bin")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

by_pdg = lambda arr: {pdg: arr[pdg_all == pdg] for pdg in particle_types}

spread_hist(by_pdg(E_mc), E_edges_5,
            "True Energy [GeV]", "Spread of True Energy", "spread_energy.png")
spread_hist(by_pdg(Angle_mc), A_edges_5,
            "True Angle [deg]", "Spread of True Angle", "spread_angle.png")
