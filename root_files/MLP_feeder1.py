#This will be for the MLP

#######################NOTES#############################
'''
-I will probably want to k-fold as I don't have enough data
-First train pions and electrons separately, then evenly together
-Use optuna to optimize things and variables
-First throw a lot of variables then reduce
-Try to have the final variable be response?
'''
######################IMPORTS#############################
import os
import glob
import math
import random
import copy

import numpy as np
#import pandas as pd

import uproot
import awkward as ak

import matplotlib.pyplot as plt

'''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
'''
####CONFIG##############################
SEED = 42
TRAIN_FRAC = 0.8
USE_CACHE = False          # True: skip extraction and load features_cache.npz
CACHE_FILE = "features_cache.npz"

# Keys stored per particle type (both in the cache and in the pooled data)
DATA_KEYS = ["X", "response", "mc_energy", "event_ids", "file_idx", "theta"]

FEATURE_NAMES = [
    "all_iso", "cog_z", "radial_first", "aspect_ratio", "radial_rms",
    "time_center", "HAD_fraction", "max_hit_had", "core_signal_frac",
    "hcal_over_all", "ecal_over_all", "z_3_alpha_1",
]

#####################Moments Available#######################
'''
Bary Eta, Bary Phi, Bary X, Bary Y, * Bary Z, Long first, * Radial first,
* Calo face first, Zeroth moment, * Aspect Ratio, Effective Eta, Effective Phi,
Theta RMS, * Radial RMS, Long RMS, Long Skew, Long Kurtosis, Radial Skew,
Radial Kurtosis, * Time Median, Energy Weighted Median, Time Moment 2/3/4,
EM Fraction, * HAD Fraction, Most Energetic All, * Most Energetic HAD,
* Core Signal Frac all/HAD/EM, z_1_alpha_0_5, z_2_alpha_1, z_2_alpha_2,
* z_3_alpha_1, z_3_alpha_2, All Hit Isolation, Cluster Isolation, Isolation Fraction
'''
########################Colors############################
styles = {
    (11,  "unc"): dict(color="deepskyblue", marker="o", dx=-1.5),
    (11,  "cal"): dict(color="navy",        marker="s", dx=-0.5),
    (211, "unc"): dict(color="orange",      marker="o", dx= 0.5),
    (211, "cal"): dict(color="firebrick",   marker="s", dx= 1.5),
}
#####################Getting_Data_and_Combining##############

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}

loc2sys = {
    "EcalBarrel": "EcalBarrelCollectionRec",
    "HcalBarrel": "HcalBarrelCollectionRec",
    "EcalEndcap": "EcalEndcapCollectionRec",
    "HcalEndcap": "HcalEndcapCollectionRec",
}

real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec", "EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]

base_path   = "/users/rldohert/data/mucoll/rldohert"
base_prompt = "/users/rldohert/work/mucoll/mucoll-slurm/files/prompt_records"
TAIL = "pt_10_200_theta_45-135"

def root_files(pdg, calib):
    """ROOT files for one calib version: 1 file for electrons, 3 parts for pions."""
    d = f"{base_path}/pdg_{pdg}_{TAIL}_{calib}"
    if pdg == 11:
        return [f"{d}/reco_pdg_11_{TAIL}_nobib.root"]
    return [f"{d}/reco_pdg_211_{TAIL}_nobib_part{p}.root" for p in (1, 2, 3)]

def npz_for(root_path):
    """Same naming as out_path() in the collect script: <parent_dir>__<file_stem>.npz"""
    tag = (os.path.basename(os.path.dirname(root_path)) + "__"
           + os.path.splitext(os.path.basename(root_path))[0])
    return f"{base_prompt}/{tag}.npz"

_files = {
    11:  root_files(11, "calib")  + root_files(11, "calib2"),    # 2 files
    211: root_files(211, "calib2") + root_files(211, "calib3"),  # 6 files
}

particles = {
    11: {
        "name": "electron", "color": "green",
        "files": _files[11],
        "prompt": [npz_for(f) for f in _files[11]],
        "test_width": 20, "matching_bound": 0.03639618,
    },
    211: {
        "name": "pion", "color": "blue",
        "files": _files[211],
        "prompt": [npz_for(f) for f in _files[211]],
        "test_width": 20, "matching_bound": 0.01243161,
    },
}

def zeroth_moment(i, j): #event, cluster
    n_clusters = len(cluster_energy[i])
    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    hit_index      = hit_index_all[i]
    collection_ID  = collectionID_all[i]

    start = hits_begin_arr[j]
    end = hits_end_arr[j]
    indices = hit_index[start:end]
    ids = collection_ID[start:end]
    if end - start == 0:
        print(f"Warning, no hits in event {i}, cluster {j}")
        return None

    #Skipping the muon system
    valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
    valid_idx = indices[valid_mask]
    valid_ids = ids[valid_mask]
    ws = np.zeros(len(valid_idx))
    if not np.any(valid_mask):
        return None  # skip cluster if all hits are "Skip"
    for sysid in np.unique(ids[valid_mask]):
        if sysid not in system2name:
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]
        ws[mask] = ener[sysname][i][idxs]

    finite_mask = (np.isfinite(ws))
    ws = ws[finite_mask]
    E_tot = np.sum(ws)
    return E_tot

def return_EM_total_HAD_total(i, j):
    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    hit_index      = hit_index_all[i]
    collection_ID  = collectionID_all[i]

    start = hits_begin_arr[j]
    end = hits_end_arr[j]
    indices = hit_index[start:end]
    ids = collection_ID[start:end]
    if end - start == 0:
        print(f"Warning, no hits in event {i}, cluster {j}")
        return None

    valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
    valid_idx = indices[valid_mask]
    valid_ids = ids[valid_mask]

    EM_total = 0.0
    HAD_total = 0.0
    for sysid in np.unique(valid_ids):
        if sysid not in system2name:
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]

        if sysname in ("EcalBarrelCollectionRec", "EcalEndcapCollectionRec"):
            EM_total += np.sum(ener[sysname][i][idxs])
        elif sysname in ("HcalBarrelCollectionRec", "HcalEndcapCollectionRec"):
            HAD_total += np.sum(ener[sysname][i][idxs])

    return EM_total, HAD_total

def return_max_hit(i, j):
    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    hit_index      = hit_index_all[i]
    collection_ID  = collectionID_all[i]

    start = hits_begin_arr[j]
    end = hits_end_arr[j]
    indices = hit_index[start:end]
    ids = collection_ID[start:end]
    if end - start == 0:
        print(f"Warning, no hits in event {i}, cluster {j}")
        return None

    valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
    valid_idx = indices[valid_mask]
    valid_ids = ids[valid_mask]
    max_hit_all = 0
    max_hit_em = 0
    max_hit_had = 0

    for sysid in np.unique(valid_ids):
        if sysid not in system2name:
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]
        max_temp = np.max(ener[sysname][i][idxs])
        if max_temp > max_hit_all:
            max_hit_all = max_temp
        if sysname in ("EcalBarrelCollectionRec", "EcalEndcapCollectionRec"):
            if max_temp > max_hit_em:
                max_hit_em = max_temp
        if sysname in ("HcalBarrelCollectionRec", "HcalEndcapCollectionRec"):
            if max_temp > max_hit_had:
                max_hit_had = max_temp
    return max_hit_all, max_hit_em, max_hit_had

def return_barycenter_centre_of_gravity(i, j): #event, cluster
    n_clusters = len(cluster_energy[i])
    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    hit_index      = hit_index_all[i]
    collection_ID  = collectionID_all[i]

    start = hits_begin_arr[j]
    end = hits_end_arr[j]
    indices = hit_index[start:end]
    ids = collection_ID[start:end]
    if end - start == 0:
        print(f"Warning, no hits in event {i}, cluster {j}")
        return None

    valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
    valid_idx = indices[valid_mask]
    valid_ids = ids[valid_mask]

    xs = np.zeros(len(valid_idx))
    ys = np.zeros(len(valid_idx))
    zs = np.zeros(len(valid_idx))
    ws = np.zeros(len(valid_idx))
    id_cell = np.zeros(len(valid_idx), dtype=np.uint64)
    locs = np.empty(len(valid_idx), dtype=object)
    if not np.any(valid_mask):
        return None  # skip cluster if all hits are "Skip"

    for sysid in np.unique(ids[valid_mask]):
        if sysid not in system2name:
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]
        xs[mask] = pos[sysname]["x"][i][idxs]
        ys[mask] = pos[sysname]["y"][i][idxs]
        zs[mask] = pos[sysname]["z"][i][idxs]
        ws[mask] = ener[sysname][i][idxs]
        id_cell[mask] = cellid[sysname][i][idxs]
        locs[mask] = location[sysname][i][idxs]

    finite_mask = (
        np.isfinite(xs)
        & np.isfinite(ys)
        & np.isfinite(zs)
        & np.isfinite(ws)
    )
    xs = xs[finite_mask]
    ys = ys[finite_mask]
    zs = zs[finite_mask]
    ws = ws[finite_mask]
    id_cell = id_cell[finite_mask]
    locations = locs[finite_mask]

    r3 = np.sqrt(xs**2 + ys**2 + zs**2)
    rho = np.sqrt(xs**2 + ys**2)
    phi = np.arctan2(ys, xs)

    theta = np.arctan2(rho, zs)
    eta = -np.log(np.tan(theta / 2.0))

    E_tot = np.sum(ws)
    if E_tot == 0:
        print(f"Warning: zero total energy in event {i}, cluster {j}")
        return None
    eta_bary = np.sum(ws * eta) / E_tot
    phi_bary = np.arctan2(np.sum(ws * np.sin(phi)), np.sum(ws * np.cos(phi)))

    cog_x = np.sum(ws * xs) / E_tot
    cog_y = np.sum(ws * ys) / E_tot
    cog_z = np.sum(ws * zs) / E_tot
    return eta_bary, phi_bary, [cog_x, cog_y, cog_z], xs, ys, zs, ws, id_cell, locations

def return_time_median_enertime_median(i, j):
    n_clusters = len(cluster_energy[i])
    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    hit_index      = hit_index_all[i]
    collection_ID  = collectionID_all[i]

    start = hits_begin_arr[j]
    end = hits_end_arr[j]
    indices = hit_index[start:end]
    ids = collection_ID[start:end]
    if end - start == 0:
        print(f"Warning, no hits in event {i}, cluster {j}")
        return None

    valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
    valid_idx = indices[valid_mask]
    valid_ids = ids[valid_mask]

    xs = np.zeros(len(valid_idx))
    ys = np.zeros(len(valid_idx))
    zs = np.zeros(len(valid_idx))
    ts = np.zeros(len(valid_idx))
    es = np.zeros(len(valid_idx))
    if not np.any(valid_mask):
        return None  # skip cluster if all hits are "Skip"

    for sysid in np.unique(ids[valid_mask]):
        if sysid not in system2name:
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]
        xs[mask] = pos[sysname]["x"][i][idxs]
        ys[mask] = pos[sysname]["y"][i][idxs]
        zs[mask] = pos[sysname]["z"][i][idxs]
        ts[mask] = time[sysname][i][idxs]
        es[mask] = ener[sysname][i][idxs]

    finite_mask = (
        np.isfinite(xs)
        & np.isfinite(ys)
        & np.isfinite(zs)
        & np.isfinite(ts)
        & np.isfinite(es)
    )
    xs = xs[finite_mask]
    ys = ys[finite_mask]
    zs = zs[finite_mask]
    ts = ts[finite_mask]
    es = es[finite_mask]

    time_center = np.mean(ts)
    energy_total = np.sum(es)
    weighted_time = np.sum(ts*es) / energy_total
    weighted_time_center = np.mean(weighted_time)

    return time_center, weighted_time_center, ts, es

def return_time_moments(ts, es, weighted_time_center, degree):
    #degree = 2 is second moment / spread (time^2)
    #degree = 3 is third moment / skewness / assymetry
    #degree = 4 is fourth moment / tails
    total_e = np.sum(es)
    time_moment = np.sum(es*(ts-weighted_time_center)**degree) / total_e
    time_moment_sr = np.sqrt(time_moment) / total_e
    return time_moment, time_moment_sr

def get_shower_axis(xs, ys, zs, ws, cog):
    pos = np.vstack([xs, ys, zs]).T
    cog = np.array(cog)
    diffs = pos - cog #spatial differences
    W = np.sum(ws)
    inertia_tensor = np.einsum('ni,nj,n->ij', diffs, diffs, ws) / W
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    axis = eigvecs[:, -1]
    if np.dot(axis, cog) < 0:
        axis = -axis
    sigma_L2 = eigvals[-1]
    sigma_T2 = eigvals[0] + eigvals[1]
    sigma_Lsq = np.sqrt(eigvals[-1])
    sigma_Tsq = np.sqrt(eigvals[0] + eigvals[1])

    theta_rms = np.arctan2(sigma_Tsq, sigma_Lsq)

    moment_ratio = sigma_T2 / sigma_L2
    aspect_ratio = sigma_Tsq / sigma_Lsq
    return axis, sigma_L2, sigma_T2, sigma_Lsq, sigma_Tsq, theta_rms, moment_ratio, aspect_ratio

def get_radial_long_dist_to_showeraxis(xs, ys, zs, cog, axis):
    pos = np.vstack([xs, ys, zs]).T
    cog = np.array(cog)
    diffs = pos - cog
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis) #normalizing

    lambdas = np.dot(diffs, axis)
    cross_products = np.cross(diffs, axis)
    radials = np.linalg.norm(cross_products, axis = 1)
    return lambdas, radials

def get_rms_cluster_dimensions(xs, ys, zs, ws, cog, axis, x_entry = None):
    lambdas, radials = get_radial_long_dist_to_showeraxis(xs, ys, zs, cog, axis)
    W = np.sum(ws)
    rsq_avg = np.sum(ws * (radials**2)) / W
    lambdasq_avg = np.sum(ws * (lambdas**2)) / W
    rms_radial = np.sqrt(rsq_avg)
    rms_longitudinal = np.sqrt(lambdasq_avg)
    r_firstmoment = np.sum(ws * (radials)) / W
    l_firstmoment = np.sum(ws * (lambdas)) / W #should be zero

    if x_entry is not None: #only possible if has a track aka charged
        lambdas_entry, _ = get_radial_long_dist_to_showeraxis(xs, ys, zs, x_entry, axis)
        l_firstmoment_entry = np.sum(ws * lambdas_entry) / W
    else:
        l_firstmoment_entry = None
    return rms_radial, rms_longitudinal, r_firstmoment, l_firstmoment, l_firstmoment_entry

def get_third_fourth_moments(xs, ys, zs, ws, cog, axis):
    lambdas, radials = get_radial_long_dist_to_showeraxis(xs, ys, zs, cog, axis)
    W = np.sum(ws)
    r_secondmoment = np.sum(ws * (radials**2)) / W
    l_secondmoment = np.sum(ws * (lambdas**2)) / W
    r_thirdmoment = np.sum(ws * (radials**3)) / W
    l_thirdmoment = np.sum(ws * (lambdas**3)) / W
    r_fourthmoment = np.sum(ws * (radials**4)) / W
    l_fourthmoment = np.sum(ws * (lambdas**4)) / W
    r_skewness = r_thirdmoment / (r_secondmoment)**(3/2)
    l_skewness = l_thirdmoment / (l_secondmoment)**(3/2)
    r_kurtosis = r_fourthmoment / (r_secondmoment)**2
    l_kurtosis = l_fourthmoment / (l_secondmoment)**2

    return r_thirdmoment, l_thirdmoment, r_fourthmoment, l_fourthmoment, r_skewness, l_skewness, r_kurtosis, l_kurtosis

def get_normalized_energy_disp(xs, ys, zs, ws, cog, axis, rcore = 4, longcore = 10):
    #the values for rcore and longcore are just chosen from atlas
    lambdas, radials = get_radial_long_dist_to_showeraxis(xs, ys, zs, cog, axis)
    W = np.sum(ws)
    top2_idx = np.argsort(ws)[-2:]

    r_out = radials.copy()
    r_out[top2_idx] = 0.0
    r2_out = np.sum(ws* (r_out**2)) / W

    r_core_arr = np.zeros_like(radials)
    r_core_arr[top2_idx] = rcore
    r2_core = np.sum(ws * (r_core_arr**2)) / W

    m2_lat = r2_out / (r2_out + r2_core)

    l_out = lambdas.copy()
    l_out[top2_idx] = 0.0
    l2_out = np.sum(ws*(l_out**2)) / W

    l_core_arr = np.zeros_like(lambdas)
    l_core_arr[top2_idx] = longcore
    l2_core = np.sum(ws * (l_core_arr**2)) / W
    m2_long = l2_out / (l2_out + l2_core)

    return m2_lat, m2_long

def get_effective_cluster_size(xs, ys, zs, ws, cog, axis, eta_bary, phi_bary):
    cog = np.array(cog)
    c_norm = np.linalg.norm(cog)
    if c_norm == 0:
        return 0.0
    rms_radial = get_rms_cluster_dimensions(xs, ys, zs, ws, cog, axis)[0]
    sigma_eta_phi = np.arctan(rms_radial / c_norm) * np.cosh(eta_bary)
    return sigma_eta_phi

def get_entry_point_from_track(event_i, track_index):
    sb = pandora_track_statesbegin[event_i][track_index]
    se = pandora_track_statesend[event_i][track_index]

    for s in range(sb, se):
        # Check if this state corresponds to AtCalorimeter (location 4)
        if track_states_location[event_i][s] == 4:
            x_entry = track_states_x[event_i][s]
            y_entry = track_states_y[event_i][s]
            z_entry = track_states_z[event_i][s]
            return x_entry, y_entry, z_entry

    return None  # Location 4 was not stored for this track

def jet_angularity(xs, ys, zs, ws, eta_c, phi_c, axis, kappa = 1.0, alpha = 1.0, R = 1.0):
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    ws = np.array(ws)

    rho = np.sqrt(xs**2 + ys**2)
    theta = np.arctan2(rho, zs)
    eta = -np.log(np.tan(theta / 2.0))
    phi = np.arctan2(ys, xs)

    deta = eta - eta_c
    dphi = np.arctan2(np.sin(phi - phi_c), np.cos(phi - phi_c))  # wrap
    deltaR = np.hypot(deta, dphi)

    theta_i = deltaR / R
    E_tot = np.sum(ws)
    z = ws / E_tot

    lam = np.sum((z ** kappa) * (theta_i ** alpha))
    return lam

def core_signal_fraction(ws, cell_ids, locs):
    ws = np.asarray(ws, dtype=np.float64)
    total_energy = np.sum(ws)
    locs = np.asarray(locs)
    ids = np.asarray(cell_ids).astype(np.uint64)
    layer = (ids >> np.uint64(19)) & np.uint64(0x1FF)

    detector_masks = [
        locs == "EcalBarrel",
        locs == "EcalEndcap",
        locs == "HcalBarrel",
        locs == "HcalEndcap",
    ]

    energy_total = 0.0
    for mask in detector_masks:
        sub_layers = layer[mask]
        sub_ws = ws[mask]
        for value in np.unique(sub_layers):
            layer_mask = sub_layers == value
            energy_total += np.max(sub_ws[layer_mask])

    return energy_total / total_energy

def isolation_allhits_but(unclaimed, cluster_idx, event_index, max_R=False, min_radius=6.0):
    """
    Per (collection, layer) the cluster touches, weighted by the cluster's energy there:
      radius = energy-weighted RMS width of the cluster's hits in that layer / 2
               (floored at min_radius if max_R is True)
      neighbours = hits in that layer NOT in this cluster within `radius` of any cluster hit
    Returns (iso_all, iso_clus):
      iso_all  = E-weighted mean number of neighbours
      iso_clus = same, but only neighbours that belong to some other cluster
    """
    start = hits_begin_all[event_index][cluster_idx]
    end = hits_end_all[event_index][cluster_idx]
    gidx = np.asarray(hit_index_all[event_index][start:end]).astype(np.int64)
    sids = np.asarray(collectionID_all[event_index][start:end])

    own = {name: set() for name in real_systems}
    for sid, g in zip(sids, gidx):
        name = system2name.get(int(sid))
        if name in own:
            own[name].add(int(g))

    num_all = den_all = num_clus = den_clus = 0.0
    for name in real_systems:
        if not own[name]:
            continue
        e = np.asarray(ener[name][event_index], dtype=float)
        px = np.asarray(pos[name]["x"][event_index], dtype=float)
        py = np.asarray(pos[name]["y"][event_index], dtype=float)
        pz = np.asarray(pos[name]["z"][event_index], dtype=float)
        cid = np.asarray(cellid[name][event_index]).astype(np.uint64)
        layer = ((cid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)

        own_arr = np.fromiter(own[name], dtype=np.int64)
        in_own = np.zeros(len(e), dtype=bool)
        in_own[own_arr] = True
        unclaimed_set = set(unclaimed[name])

        for lay in np.unique(layer[own_arr]):
            in_layer = layer == lay
            our_idx = np.nonzero(in_layer & in_own)[0]
            cand_idx = np.nonzero(in_layer & ~in_own)[0]
            if len(cand_idx) == 0 or len(our_idx) == 0:
                continue

            w = e[our_idx]
            E_s = w.sum()
            if not np.isfinite(E_s) or E_s <= 0:
                continue
            ox, oy, oz = px[our_idx], py[our_idx], pz[our_idx]
            cx, cy, cz = np.sum(w*ox)/E_s, np.sum(w*oy)/E_s, np.sum(w*oz)/E_s
            width = np.sqrt(np.sum(w*((ox-cx)**2 + (oy-cy)**2 + (oz-cz)**2)) / E_s)
            radius = width / 2
            if max_R and radius < min_radius:
                radius = min_radius

            nx, ny, nz = px[cand_idx], py[cand_idx], pz[cand_idx]
            d2 = ((ox[:, None]-nx[None, :])**2
                  + (oy[:, None]-ny[None, :])**2
                  + (oz[:, None]-nz[None, :])**2)
            nearby_idx = cand_idx[(d2 < radius**2).any(axis=0)]
            n_neighbor = len(nearby_idx)
            n_noclus = sum(1 for g in nearby_idx if int(g) in unclaimed_set)

            num_all += E_s * n_neighbor;                 den_all += E_s
            num_clus += E_s * (n_neighbor - n_noclus);   den_clus += E_s

    iso_all = num_all / den_all if den_all > 0 else None
    iso_clus = num_clus / den_clus if den_clus > 0 else None
    return iso_all, iso_clus

def core_signal_fraction_detector(ws, cell_ids, locs):

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

    energy_total = 0.0

    count_ecal = 0
    count_hcal = 0

    energy_core_ecal = 0.0
    energy_core_hcal = 0.0

    energy_all_ecal = 0.0
    energy_all_hcal = 0.0

    for detector, mask in detector_masks.items():

        sub_layers = layer[mask]
        sub_ws = ws[mask]

        length = len(sub_layers)

        if detector in ("EcalBarrel", "EcalEndcap"):
            count_ecal += length
            energy_all_ecal += np.sum(sub_ws)

        elif detector in ("HcalBarrel", "HcalEndcap"):
            count_hcal += length
            energy_all_hcal += np.sum(sub_ws)

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

def load_calorimeter_arrays():
    pos = {}
    ener = {}
    time = {}
    cellid = {}
    location = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        pos[name] = {
            "x": events[f"{prefix}.position.x"].array(),
            "y": events[f"{prefix}.position.y"].array(),
            "z": events[f"{prefix}.position.z"].array(),
        }
        ener[name] = events[f"{prefix}.energy"].array()
        time[name] = events[f"{prefix}.time"].array()
        cellid[name] = events[f"{prefix}.cellID"].array()
        if name == "EcalBarrelCollectionRec":
            loc_name = "EcalBarrel"
        elif name == "HcalBarrelCollectionRec":
            loc_name = "HcalBarrel"
        elif name == "EcalEndcapCollectionRec":
            loc_name = "EcalEndcap"
        elif name == "HcalEndcapCollectionRec":
            loc_name = "HcalEndcap"

        location[name] = ak.Array([
            [loc_name] * len(event)
            for event in ener[name]
        ])
    return pos, ener, time, cellid, location

#We need to know if the indexes are unclaimed
#Passed in an event, passes out the indexes of the unclaimed hits for that event
def unclaimed_indexes(event_index, n_clusters):
    hits_begin_arr = hits_begin_all[event_index]
    hits_end_arr = hits_end_all[event_index]
    hit_index = hit_index_all[event_index]
    collection_ID = collectionID_all[event_index]
    claimed = {name: set() for name in real_systems}

    for j in range(n_clusters):
        start = hits_begin_arr[j]
        end = hits_end_arr[j]
        idxs = np.asarray(hit_index[start:end])
        ids = np.asarray(collection_ID[start:end])

        for sysid, idx in zip(ids, idxs):
            sysname = system2name.get(sysid, None)
            if sysname is None or sysname == "Skip":
                continue
            claimed[sysname].add(int(idx))
    unclaimed = {}
    for sysname in real_systems:
        n_hits_total = len(ener[sysname][event_index])
        all_idx = set(range(n_hits_total))
        unclaimed[sysname] = sorted(all_idx - claimed[sysname])
    return unclaimed #returns dictionary with unclaimed hits by the Ecal...Rec collection name

def cluster_matching(i, bound):
    mask = (status_mc[i] == 1)
    mx = np.asarray(mc_px[i])[mask]
    my = np.asarray(mc_py[i])[mask]
    mz = np.asarray(mc_pz[i])[mask]
    if len(mx) == 0:
        return None

    mc_r = np.sqrt(mx**2 + my**2 + mz**2)
    mc_theta = np.arccos(mz / mc_r)
    mc_phi = np.arctan2(my, mx)

    num_clusters = len(hits_begin_all[i])
    clus_ids = []
    center_x, center_y, center_z, energy = [], [], [], []
    for clus in range(num_clusters):
        res = return_barycenter_centre_of_gravity(i, clus)
        if res is None:
            continue
        _, _, (cog_x, cog_y, cog_z), _, _, _, ws, _, _ = res
        clus_ids.append(clus)
        center_x.append(cog_x)
        center_y.append(cog_y)
        center_z.append(cog_z)
        energy.append(np.sum(ws))

    if len(clus_ids) == 0:
        return None

    clus_ids = np.asarray(clus_ids)
    center_x = np.asarray(center_x)
    center_y = np.asarray(center_y)
    center_z = np.asarray(center_z)
    energy = np.asarray(energy)

    c_r = np.sqrt(center_x**2 + center_y**2 + center_z**2)
    c_theta = np.arccos(center_z / c_r)
    c_phi = np.arctan2(center_y, center_x)

    cosang = (np.sin(c_theta)[:, None] * np.sin(mc_theta)[None, :]
              * np.cos(c_phi[:, None] - mc_phi[None, :])
              + np.cos(c_theta)[:, None] * np.cos(mc_theta)[None, :])
    cosang = np.clip(cosang, -1.0, 1.0)
    angular_distance = np.arccos(cosang)

    is_candidate = (angular_distance <= bound).any(axis=1)
    if not is_candidate.any():
        return None

    cand = np.nonzero(is_candidate)[0]
    best = cand[np.argmax(energy[cand])]
    return int(clus_ids[best])      # fixed: trailing comma used to make this a 1-tuple

def cluster_response(event, cluster_energy):
    """E_cluster / E_true and E_true. E_true sums final-state MC particles."""
    if cluster_energy is None:
        return np.nan, np.nan          # fixed: used to return a bare np.nan
    mask = np.asarray(status_mc[event]) == 1
    px = np.asarray(mc_px[event], dtype=float)[mask]
    py = np.asarray(mc_py[event], dtype=float)[mask]
    pz = np.asarray(mc_pz[event], dtype=float)[mask]
    m = np.asarray(mc_mass[event], dtype=float)[mask]
    mc_energy = np.sum(np.sqrt(m**2 + px**2 + py**2 + pz**2))
    if mc_energy <= 0:
        return np.nan, np.nan
    return cluster_energy / mc_energy, mc_energy

def extract_angles(event):
    """Polar angle (radians) of each final-state MC particle in the event."""
    mask = np.asarray(status_mc[event]) == 1
    px = np.asarray(mc_px[event], dtype=float)[mask]
    py = np.asarray(mc_py[event], dtype=float)[mask]
    pz = np.asarray(mc_pz[event], dtype=float)[mask]
    p   = np.sqrt(px**2 + py**2 + pz**2)
    pt  = np.sqrt(px**2 + py**2)
    theta = np.arccos( pz / p )
    return theta

def event_theta_deg(event):
    """Scalar true polar angle in degrees for this event (nan if no final-state particle)."""
    th = extract_angles(event)
    if len(th) == 0:
        return np.nan
    return float(np.degrees(th[0]))


#Heads up what the .npz contains
#pdg, event, cluster, tag, n_clusters, calib_prompt, iso_all, event_prompt_sum, mc_energy, theta_deg 
######################################Filling################################
all_particle_data = {}
import sys
task = int(sys.argv[1]) if len(sys.argv) > 1 else None   # None = run everything

OUT_DIR = "features_per_file1"
os.makedirs(OUT_DIR, exist_ok=True)

n = -1
for pdg, info in particles.items():
    for file_idx, (root_path, npz_path) in enumerate(zip(info["files"], info["prompt"])):
        n += 1
        if task is not None and n != task:
            continue

        out_file = os.path.join(
            OUT_DIR, f"features_pdg{pdg}_{os.path.basename(os.path.dirname(root_path))}"
                     f"__{os.path.splitext(os.path.basename(root_path))[0]}.npz")
        print(f"Processing {root_path}", flush=True)

        # ---- npz ----
        d = np.load(npz_path)
        rec_tag    = d["tag"]
        rec_prompt = d["calib_prompt"]
        rec_iso    = d["iso_all"]
        rec_evsum  = d["event_prompt_sum"]
        rec_mc     = d["mc_energy"]
        rec_theta  = d["theta_deg"]
        lookup = {(int(e), int(c)): i
                  for i, (e, c) in enumerate(zip(d["event"], d["cluster"]))}
        '''
        d = np.load(npz_path)
        rec = pd.DataFrame({k: d[k] for k in d.keys()})
        '''

        # ---- root ----
        file = uproot.open(root_path)
        events = file["events"]
        n_ev = events.num_entries
        pandora_clusters = events["PandoraClusters"]
        pandora_clusters_hits = events["_PandoraClusters_hits"]
        cluster_energy = pandora_clusters["PandoraClusters.energy"].array()
        hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
        hits_end_all = pandora_clusters["PandoraClusters.hits_end"].array()
        hit_index_all = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
        collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
        pandora_clusters_begin = events["PandoraPFOs/PandoraPFOs.clusters_begin"].array()
        pandora_clusters_end = events["PandoraPFOs/PandoraPFOs.clusters_end"].array()
        pandora_clusters_index = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"].array()
        pandora_tracks_begin = events["PandoraPFOs/PandoraPFOs.tracks_begin"].array()
        pandora_tracks_end = events["PandoraPFOs/PandoraPFOs.tracks_end"].array()
        track_states_x = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.x"].array()
        track_states_y = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.y"].array()
        track_states_z = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.z"].array()
        track_states_location = events["_AllTracks_trackStates/_AllTracks_trackStates.location"].array()
        pandora_track_statesbegin = events["AllTracks/AllTracks.trackStates_begin"].array()
        pandora_track_statesend = events["AllTracks/AllTracks.trackStates_end"].array()
        pandora_tracks_index = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"].array()
        pos, ener, time, cellid, location = load_calorimeter_arrays()
        pfo_energy = events["PandoraPFOs/PandoraPFOs.energy"].array()
        mcparticles = events["MCParticles"]
        #status_mc = mcparticles["MCParticles.generatorStatus"].array()
        #mc_px = mcparticles["MCParticles.momentum.x"].array()
        #mc_py = mcparticles["MCParticles.momentum.y"].array()
        #mc_pz = mcparticles["MCParticles.momentum.z"].array()
        #mc_mass = mcparticles["MCParticles.mass"].array()

        event_ids, cluster_ids = [], []
        cog_z_l, radial_first_l, aspect_l, radial_rms_l = [], [], [], []
        time_center_l, core_signal_l, hcal_over_all_l, ecal_over_all_l, z3a1_l = [], [], [], [], []
        had_frac_l, max_had_l, all_iso_l = [], [], []
        response_l, mc_energy_l, theta_l = [], [], []
        tag_l, calib_l, evsum_l = [], [], []
        for index in range(n_ev):
            #unclaimed = unclaimed_indexes(index, len(cluster_energy[index]))

            #Normal check for 0, excluding now
            #if len(pfo_energy[index]) != 1:
            #    continue
            #
            if True:
                n_cl = len(hits_begin_all[index])
                if n_cl == 0:
                    continue

                # entry point per cluster, from the PFO that owns it (if any)
                entry_for_cluster = {}
                for pfo in range(len(pfo_energy[index])):
                    tracks_begin = pandora_tracks_begin[index][pfo]
                    tracks_end = pandora_tracks_end[index][pfo]
                    entry_point = None
                    for track in range(tracks_begin, tracks_end):
                        track_index = pandora_tracks_index[index][track]
                        entry = get_entry_point_from_track(index, track_index)
                        if entry is not None:
                            entry_point = list(entry)
                    for c in range(pandora_clusters_begin[index][pfo], pandora_clusters_end[index][pfo]):
                        entry_for_cluster[int(pandora_clusters_index[index][c])] = entry_point

                for cluster_index in range(n_cl):          # ALL clusters in the event
                    entry_point = entry_for_cluster.get(cluster_index)   # None if no PFO/track
                    row = lookup.get((index, cluster_index))
                    if row is None:
                        continue
                    #check if cluster tag = 1, if its 1 continue
                    if rec_tag[row] != 0:
                        continue   # tag 1 = max-iso cluster, skip it
                    calib_prompt = rec_prompt[row]
                    mc_energy    = rec_mc[row]
                    theta        = rec_theta[row]

                    total_energy = zeroth_moment(index, cluster_index)
                    #response, mc_energy = cluster_response(index, total_energy)
                    res = return_barycenter_centre_of_gravity(index, cluster_index)
                    if total_energy is None or res is None or not calib_prompt > 0:
                        continue
                    response = total_energy / calib_prompt
                    eta_bary, phi_bary, [cog_x, cog_y, cog_z], xs, ys, zs, ws, id_cell, locs = res
                    cog = [cog_x, cog_y, cog_z]

                    core_signal_frac = core_signal_fraction(ws, id_cell, locs)
                    (_core2, _hcalratio, hcal_over_all, ecal_over_all,
                     _hh, _ee) = core_signal_fraction_detector(ws, id_cell, locs)

                    axis, _sL2, _sT2, _sLsq, sigma_T2sq, _theta_rms, _mr, aspect_ratio = \
                        get_shower_axis(xs, ys, zs, ws, cog)

                    '''
                    iso_all, iso_clus = isolation_allhits_but(
                        unclaimed, cluster_index, index, max_R=(sigma_T2sq <= 12))
                    if iso_all is None or iso_clus is None:
                        iso_all, iso_clus, iso_frac = np.nan, np.nan, np.nan
                    elif iso_all == 0:
                        iso_frac = -1
                    else:
                        iso_frac = iso_clus / iso_all
                    '''
                    rms_radial, _rmsL, r_firstmoment, _l1, l_firstmoment_entry = \
                        get_rms_cluster_dimensions(xs, ys, zs, ws, cog, axis, entry_point)

                    #m2_lat, m2_long = get_normalized_energy_disp(xs, ys, zs, ws, cog, axis)
                    #sigma_eta_phi = get_effective_cluster_size(xs, ys, zs, ws, cog, axis, eta_bary, phi_bary)
                    #(r_thirdmoment, l_thirdmoment, r_fourthmoment, l_fourthmoment,
                    # r_skewness, l_skewness, r_kurtosis, l_kurtosis) = get_third_fourth_moments(xs, ys, zs, ws, cog, axis)

                    time_center, _ewc, _ts, _es = return_time_median_enertime_median(index, cluster_index)
                    #time_moment2, _ = return_time_moments(_ts, _es, _ewc, 2)
                    #time_moment3, _ = return_time_moments(_ts, _es, _ewc, 3)
                    #time_moment4, _ = return_time_moments(_ts, _es, _ewc, 4)

                    #k_1_a_0_5 = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa=1.0, alpha=0.5, R=1.0)
                    #k_2_a_1   = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa=2.0, alpha=1.0, R=1.0)
                    #k_2_a_2   = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa=2.0, alpha=2.0, R=1.0)
                    k_3_a_1 = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa=3.0, alpha=1.0, R=1.0)
                    #k_3_a_2   = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa=3.0, alpha=2.0, R=1.0)

                    # HAD fraction and max HAD hit (same loop so rows stay aligned)
                    _em_total, had_total = return_EM_total_HAD_total(index, cluster_index)
                    _max_all, _max_em, max_hit_had = return_max_hit(index, cluster_index)
                    #em_frac = _em_total / total_energy
                    #max_frac = _max_all / total_energy
                    #max_frac_em = _max_em / total_energy

                    event_ids.append(index)
                    cluster_ids.append(cluster_index)
                    #clus_iso_l.append(iso_clus); iso_frac_l.append(iso_frac)
                    all_iso_l.append(rec_iso[row])
                    cog_z_l.append(cog_z)
                    radial_first_l.append(r_firstmoment)
                    #longcalo_first_l.append(-999.0 if l_firstmoment_entry is None else l_firstmoment_entry)
                    #longcalo_first_l.append(np.nan if l_firstmoment_entry is None else l_firstmoment_entry)
                    aspect_l.append(aspect_ratio)
                    radial_rms_l.append(rms_radial)
                    time_center_l.append(time_center)
                    had_frac_l.append(had_total / total_energy)
                    max_had_l.append(max_hit_had / total_energy)
                    core_signal_l.append(core_signal_frac)
                    hcal_over_all_l.append(hcal_over_all)
                    ecal_over_all_l.append(ecal_over_all)
                    z3a1_l.append(k_3_a_1)
                    response_l.append(response)
                    mc_energy_l.append(mc_energy)
                    theta_l.append(theta)
                    tag_l.append(rec_tag[row])
                    calib_l.append(calib_prompt)
                    evsum_l.append(rec_evsum[row])

        columns = {
            #"clus_iso": clus_iso_l, "iso_fraction": iso_frac_l,
            "all_iso": all_iso_l,
            "cog_z": cog_z_l, "radial_first": radial_first_l,
            "aspect_ratio": aspect_l, "radial_rms": radial_rms_l, "time_center": time_center_l,
            "HAD_fraction": had_frac_l, "max_hit_had": max_had_l, "core_signal_frac": core_signal_l,
            "hcal_over_all": hcal_over_all_l, "ecal_over_all": ecal_over_all_l, "z_3_alpha_1": z3a1_l,
        }
        X_particle = np.column_stack([np.asarray(v, dtype=float) for v in columns.values()])
        response = np.asarray(response_l, dtype=float)
        mc_en = np.asarray(mc_energy_l, dtype=float)
        thetas = np.asarray(theta_l, dtype=float)

        good = (np.isfinite(X_particle).all(axis=1) & np.isfinite(response)
                & np.isfinite(mc_en) & np.isfinite(thetas))
        print(f"  Keeping {good.sum()} / {len(good)} clusters", flush=True)

        # ---- one output file per input file ----
        np.savez(out_file,
                 X=X_particle[good],
                 response=response[good],
                 mc_energy=mc_en[good],
                 theta=thetas[good],
                 event_ids=np.asarray(event_ids)[good],
                 cluster_ids=np.asarray(cluster_ids)[good],
                 tag=np.asarray(tag_l)[good],
                 calib_prompt=np.asarray(calib_l)[good],
                 event_prompt_sum=np.asarray(evsum_l)[good],
                 pdg=np.full(good.sum(), pdg),
                 file_idx=np.full(good.sum(), file_idx),
                 feature_names=np.array(list(columns.keys())))
        print(f"  saved -> {out_file}", flush=True)