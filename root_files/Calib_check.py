# =========================
# IMPORTS
# =========================
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")          # no display needed on cluster
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CALIB_DETECTORS = [
    "ECalBarrelCollection", "ECalEndcapCollection",
    "HCalBarrelCollection", "HCalEndcapCollection",
]
CALIB_SUFFIXES = [
    "Calib", "CalibEM", "CalibInvisible",
    "CalibPrompt", "CalibEMPrompt", "CalibInvisiblePrompt",
]
calib_collections = [f"{d}{s}" for d in CALIB_DETECTORS for s in CALIB_SUFFIXES]

system2name = {
    679272617:  "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = [
    "EcalBarrelCollectionRec", "HcalBarrelCollectionRec",
    "EcalEndcapCollectionRec", "HcalEndcapCollectionRec",
]

det_map = {
    "EcalBarrelCollectionRec": "ECalBarrelCollection",
    "HcalBarrelCollectionRec": "HCalBarrelCollection",
    "EcalEndcapCollectionRec": "ECalEndcapCollection",
    "HcalEndcapCollectionRec": "HCalEndcapCollection",
}

BASE = "/users/rldohert/data/mucoll/rldohert"

particles = {
    11: {
        "label": "Electron",
        "color": "deepskyblue",
        "files": [
            f"{BASE}/pdg_11_pt_10_200_theta_45-135_calib/reco_pdg_11_pt_10_200_theta_45-135_nobib.root",
        ],
    },
    211: {
        "label": "Pion",
        "color": "firebrick",
        "files": [
            f"{BASE}/pdg_211_pt_10_200_theta_45-135_calib/reco_pdg_211_pt_10_200_theta_45-135_nobib.root",
        ],
    },
}

THETA_BINS  = [(45, 63), (63, 81), (81, 99), (99, 117), (117, 135)]
ENERGY_BINS = [(10, 50), (50, 90), (90, 130), (130, 170), (170, 200)]

# =========================
# HELPERS
# =========================

def load_calorimeter_arrays(events):
    """Load only energy and cellID for reco collections — drop positions and time."""
    ener   = {}
    cellid = {}
    for name in real_systems:
        prefix       = f"{name}/{name}"
        ener[name]   = events[f"{prefix}.energy"].array()
        cellid[name] = events[f"{prefix}.cellID"].array()
    return ener, cellid


def load_calib_arrays(events):
    """Load only energy and cellID for calib collections — no positions."""
    calib_ener   = {}
    calib_cellid = {}
    for name in calib_collections:
        prefix = f"{name}/{name}"
        try:
            calib_ener[name]   = events[f"{prefix}.energy"].array()
            calib_cellid[name] = events[f"{prefix}.cellID"].array()
        except Exception as e:
            print(f"Warning: could not load calib collection {name}: {e}")
            calib_ener[name] = calib_cellid[name] = None
    return calib_ener, calib_cellid


def build_calib_lookup(event_index, calib_ener, calib_cellid):
    """
    Build a dict-of-dicts once per event: {collection_name: {cellID: energy}}.
    Reused for every cluster in the event — far cheaper than np.isin per cluster.
    """
    lookup = {}
    for name in calib_collections:
        if calib_ener[name] is None:
            continue
        cids = np.asarray(calib_cellid[name][event_index])
        ens  = np.asarray(calib_ener[name][event_index])
        lookup[name] = dict(zip(cids.tolist(), ens.tolist()))
    return lookup


def zeroth_moment(index, cluster_index, ener, cellid,
                  hits_begin_all, hits_end_all, hit_index_all, collectionID_all):
    """Sum of hit energies in the cluster, skipping muon-system hits."""
    start   = int(hits_begin_all[index][cluster_index])
    end     = int(hits_end_all[index][cluster_index])
    if end == start:
        return None
    indices = np.asarray(hit_index_all[index][start:end])
    ids     = np.asarray(collectionID_all[index][start:end])

    total = 0.0
    for sysid in np.unique(ids):
        sysname = system2name.get(int(sysid))
        if sysname is None or sysname == "Skip":
            continue
        mask = ids == sysid
        es   = np.asarray(ener[sysname][index])[indices[mask]]
        total += float(np.nansum(es))
    return total if total > 0 else None


def match_energy_fast(index, cluster_index, ener, cellid,
                      hits_begin_all, hits_end_all, hit_index_all, collectionID_all,
                      calib_lookup):
    """
    Match cluster hits to calib collections via pre-built cellID lookup.
    Returns (Calib, CalibEM, CalibInvisible, CalibPrompt, CalibEMPrompt, CalibInvisiblePrompt).
    """
    start   = int(hits_begin_all[index][cluster_index])
    end     = int(hits_end_all[index][cluster_index])
    indices = np.asarray(hit_index_all[index][start:end])
    ids     = np.asarray(collectionID_all[index][start:end])

    # collect cellIDs per calib detector
    cluster_cellids = {det: set() for det in CALIB_DETECTORS}
    for sysid, idx in zip(ids, indices):
        sysname = system2name.get(int(sysid))
        if sysname is None or sysname == "Skip":
            continue
        det = det_map.get(sysname)
        if det is None:
            continue
        cid = int(np.asarray(cellid[sysname][index])[idx])
        cluster_cellids[det].add(cid)

    results = {suf: 0.0 for suf in CALIB_SUFFIXES}
    for det in CALIB_DETECTORS:
        if not cluster_cellids[det]:
            continue
        for suf in CALIB_SUFFIXES:
            name = f"{det}{suf}"
            lut  = calib_lookup.get(name)
            if lut is None:
                continue
            for cid in cluster_cellids[det]:
                if cid in lut:
                    results[suf] += lut[cid]

    return tuple(results[s] for s in CALIB_SUFFIXES)


def event_theta_deg(index, status_mc, mc_px, mc_py, mc_pz):
    """True polar angle in degrees of the first final-state MC particle."""
    mask = np.asarray(status_mc[index]) == 1
    px   = np.asarray(mc_px[index], dtype=float)[mask]
    py   = np.asarray(mc_py[index], dtype=float)[mask]
    pz   = np.asarray(mc_pz[index], dtype=float)[mask]
    if len(px) == 0:
        return np.nan
    p     = np.sqrt(px[0]**2 + py[0]**2 + pz[0]**2)
    if p == 0:
        return np.nan
    return float(np.degrees(np.arccos(pz[0] / p)))


def mc_true_energy(index, status_mc, mc_px, mc_py, mc_pz, mc_mass):
    """Sum of true energy of all final-state MC particles."""
    mask = np.asarray(status_mc[index]) == 1
    px   = np.asarray(mc_px[index],   dtype=float)[mask]
    py   = np.asarray(mc_py[index],   dtype=float)[mask]
    pz   = np.asarray(mc_pz[index],   dtype=float)[mask]
    m    = np.asarray(mc_mass[index],  dtype=float)[mask]
    e    = float(np.sum(np.sqrt(m**2 + px**2 + py**2 + pz**2)))
    return e if e > 0 else np.nan


# =========================
# MAIN LOOP
# =========================

results_by_pdg = {}

for pdg, info in particles.items():
    new_responses  = []
    total_energies = []
    just_prompt    = []
    theta_all      = []
    mc_energy_all  = []

    for file_path in info["files"]:
        print(f"\n{'='*60}")
        print(f"Processing {info['label']} | {file_path}")
        print(f"{'='*60}")

        file   = uproot.open(file_path)
        events = file["events"]
        n_ev   = events.num_entries
        print(f"  Entries: {n_ev}")

        # -- Cluster / hit index arrays --
        cluster_energy_arr = events["PandoraClusters/PandoraClusters.energy"].array()
        hits_begin_all     = events["PandoraClusters/PandoraClusters.hits_begin"].array()
        hits_end_all       = events["PandoraClusters/PandoraClusters.hits_end"].array()
        hit_index_all      = events["_PandoraClusters_hits/_PandoraClusters_hits.index"].array()
        collectionID_all   = events["_PandoraClusters_hits/_PandoraClusters_hits.collectionID"].array()

        # -- MC arrays --
        status_mc = events["MCParticles/MCParticles.generatorStatus"].array()
        mc_px     = events["MCParticles/MCParticles.momentum.x"].array()
        mc_py     = events["MCParticles/MCParticles.momentum.y"].array()
        mc_pz     = events["MCParticles/MCParticles.momentum.z"].array()
        mc_mass   = events["MCParticles/MCParticles.mass"].array()

        # -- Calorimeter arrays (no positions, no time) --
        ener, cellid                 = load_calorimeter_arrays(events)
        calib_ener, calib_cellid     = load_calib_arrays(events)

        # -- Event loop --
        n_single = 0
        for index in range(n_ev):
            if len(cluster_energy_arr[index]) != 1:
                continue
            n_single += 1

            cluster_index = 0

            # calib lookup built once per event
            calib_lookup = build_calib_lookup(index, calib_ener, calib_cellid)

            total_energy = zeroth_moment(
                index, cluster_index, ener, cellid,
                hits_begin_all, hits_end_all, hit_index_all, collectionID_all
            )
            if total_energy is None:
                continue

            (_, _, _, calibPrompt, _, _) = match_energy_fast(
                index, cluster_index, ener, cellid,
                hits_begin_all, hits_end_all, hit_index_all, collectionID_all,
                calib_lookup
            )

            if calibPrompt <= 0:
                print(f"  Warning: calibPrompt <= 0 in event {index} "
                      f"(value={calibPrompt:.4f}), skipping.")
                continue

            theta   = event_theta_deg(index, status_mc, mc_px, mc_py, mc_pz)
            mc_ener = mc_true_energy(index, status_mc, mc_px, mc_py, mc_pz, mc_mass)

            new_responses.append(total_energy / calibPrompt)
            total_energies.append(total_energy)
            just_prompt.append(calibPrompt)
            theta_all.append(theta)
            mc_energy_all.append(mc_ener)

        print(f"  Single-cluster events processed : {n_single}")
        print(f"  Valid responses collected       : {len(new_responses)}")

        # free per-file arrays immediately
        del ener, cellid, calib_ener, calib_cellid
        del cluster_energy_arr, hits_begin_all, hits_end_all
        del hit_index_all, collectionID_all
        del status_mc, mc_px, mc_py, mc_pz, mc_mass

    results_by_pdg[pdg] = {
        "new_responses":  np.array(new_responses),
        "total_energies": np.array(total_energies),
        "just_prompt":    np.array(just_prompt),
        "theta_all":      np.array(theta_all),
        "mc_energy_all":  np.array(mc_energy_all),
    }

# =========================
# PLOT HELPER
# =========================

def save_hist(ax, data_by_pdg, key, xlabel, title, range_, bins):
    any_data = False
    for pdg, info in particles.items():
        data = data_by_pdg[pdg][key]
        if len(data) == 0:
            continue
        any_data = True
        ax.hist(data, bins=bins, range=range_, color=info["color"],
                label=f"{info['label']} (n={len(data)})",
                histtype="step", linewidth=1.8)
    if not any_data:
        return False
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    return True


plot_configs = [
    {
        "key":    "new_responses",
        "xlabel": "Response ($E_{\\mathrm{reco}}\\,/\\,E_{\\mathrm{calibPrompt}}$)",
        "title":  "Cluster Response: Electrons vs Pions\n(single-cluster events)",
        "fname":  "response_histogram.png",
        "range":  (0, 3),
        "bins":   60,
    },
    {
        "key":    "total_energies",
        "xlabel": "Total Cluster Energy [GeV]",
        "title":  "Total Cluster Energy: Electrons vs Pions\n(single-cluster events)",
        "fname":  "total_energy_histogram.png",
        "range":  (0, 250),
        "bins":   60,
    },
    {
        "key":    "just_prompt",
        "xlabel": "$E_{\\mathrm{calibPrompt}}$ [GeV]",
        "title":  "CalibPrompt Energy: Electrons vs Pions\n(single-cluster events)",
        "fname":  "calibprompt_histogram.png",
        "range":  (0, 250),
        "bins":   60,
    },
]

# =========================
# PLOT — global
# =========================

for cfg in plot_configs:
    fig, ax = plt.subplots(figsize=(8, 5))
    save_hist(ax, results_by_pdg, cfg["key"], cfg["xlabel"],
              cfg["title"], cfg["range"], cfg["bins"])
    plt.tight_layout()
    plt.savefig(cfg["fname"], dpi=150)
    plt.close(fig)
    print(f"Saved: {cfg['fname']}")

# =========================
# PLOT — binned
# =========================

for t_lo, t_hi in THETA_BINS:
    for e_lo, e_hi in ENERGY_BINS:
        for cfg in plot_configs:
            # build masked sub-dict for this bin
            binned = {}
            any_data = False
            for pdg in particles:
                r      = results_by_pdg[pdg]
                mask   = (
                    (r["theta_all"]     >= t_lo) & (r["theta_all"]     < t_hi) &
                    (r["mc_energy_all"] >= e_lo) & (r["mc_energy_all"] < e_hi)
                )
                binned[pdg] = {cfg["key"]: r[cfg["key"]][mask]}
                if mask.sum() > 0:
                    any_data = True

            if not any_data:
                continue

            title = (
                f"{cfg['key'].replace('_', ' ').title()}: Electrons vs Pions\n"
                f"$\\theta \\in [{t_lo}°,{t_hi}°)$,  "
                f"$E_{{\\mathrm{{true}}}} \\in [{e_lo},{e_hi})$ GeV"
            )
            fname = f"{cfg['key']}_theta{t_lo}-{t_hi}_E{e_lo}-{e_hi}.png"

            fig, ax = plt.subplots(figsize=(8, 5))
            save_hist(ax, binned, cfg["key"], cfg["xlabel"], title,
                      cfg["range"], cfg["bins"])
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"Saved: {fname}")