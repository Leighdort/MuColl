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
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import uproot
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
choices = [50]
import awkward as ak
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
import corner
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    log_loss
)

#####################Moments Available#######################
'''
Bary Eta
Bary Phi
Bary X
Bary Y
* Bary Z
Long first
* Radial first
* Calo face first
Zeroth moment
* Aspect Ratio
Effective Eta
Effective Phi
Theta RMS
* Radial RMS
Long RMS
Long Skew
Long Kurtosis
Radial Skew
Radial Kurtosis
* Time Median
Energy Weighted Median
Time Moment 2
Time Moment 3
Time Moment 4
EM Fraction
* HAD Fraction
Most Energetic All
* Most Energetic HAD
* Core Signal Frac all
* Core Signal Frac HAD
* Core Signal Frac EM
z_1_alpha_0_5
z_2_alpha_1
z_2_alpha_2
* z_3_alpha_1
z_3_alpha_2
All Hit Isolation
Cluster Isolation
Isolation Fraction
'''
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

real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
'''
particles = {
    11: {
        "name": "electron",
        "color": "green",
        "file": "/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_11_nobib0.edm4hep.root",
        "test_width": 20,
        "matching_bound": 0.03639618
    }
}
'''

particles = {
    211: {
        "name": "pion",
        "color": "royalblue",
        "file": "/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root",
        "test_width": 20,
        "matching_bound": 0.01243161
    },
    
    11: {
        "name": "electron",
        "color": "green",
        "file": "/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_11_nobib0.edm4hep.root",
        "test_width": 20,
        "matching_bound": 0.03639618
    }
    
}


def zeroth_moment(i, j): #event, cluster
    n_clusters = len(cluster_energy[i])
    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    hit_index      = hit_index_all[i]
    collection_ID  = collectionID_all[i]
    #Here is where a loop would go
    
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
            #print(f"Unknown id: {sysid}, from cluster {j}, from event {i}, with energy {cluster_energy[i][j]}")
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]
        #mask = (ids == sysid) & valid_mask
        #idxs = indices[mask]
        ws[mask] = ener[sysname][i][idxs]
    
    finite_mask = ( #Protects against naans 
        np.isfinite(ws)
    )
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

    # Skipping the muon system
    valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
    valid_idx = indices[valid_mask]
    valid_ids = ids[valid_mask]

    EM_total = 0.0
    HAD_total = 0.0
    for sysid in np.unique(valid_ids):
        if sysid not in system2name:
            #print(f"Unknown id: {sysid}, from cluster {j}, from event {i}")
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

    # Skipping the muon system
    valid_mask = np.array([system2name.get(sysid, None) != "Skip" for sysid in ids])
    valid_idx = indices[valid_mask]
    valid_ids = ids[valid_mask]
    max_hit_all = 0
    max_hit_em = 0
    max_hit_had = 0

    for sysid in np.unique(valid_ids):
        if sysid in np.unique(valid_ids):
            if sysid not in system2name:
                continue
        sysname = system2name[sysid]
        mask = valid_ids ==sysid
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
    #Here is where a loop would go
    
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

    xs = np.zeros(len(valid_idx))
    ys = np.zeros(len(valid_idx))
    zs = np.zeros(len(valid_idx))
    ws = np.zeros(len(valid_idx))
    id_cell = np.zeros(len(valid_idx), dtype=np.uint64)
    locs = np.empty(len(valid_idx), dtype=object)
    if not np.any(valid_mask):
        return None  # skip cluster if all hits are "Skip"

    #Filling the array
    for sysid in np.unique(ids[valid_mask]):
        if sysid not in system2name:
            #print(f"Unknown id: {sysid}, from cluster {j}, from event {i}, with energy {cluster_energy[i][j]}")
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]
        #mask = (ids == sysid) & valid_mask
        #idxs = indices[mask]
        xs[mask] = pos[sysname]["x"][i][idxs]
        ys[mask] = pos[sysname]["y"][i][idxs]
        zs[mask] = pos[sysname]["z"][i][idxs]
        ws[mask] = ener[sysname][i][idxs]
        id_cell[mask] = cellid[sysname][i][idxs]
        locs[mask] = location[sysname][i][idxs]
    
    # Making finite
    finite_mask = ( #Protects against naans 
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
    #Here is where a loop would go
    
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

    xs = np.zeros(len(valid_idx))
    ys = np.zeros(len(valid_idx))
    zs = np.zeros(len(valid_idx))
    ts = np.zeros(len(valid_idx))
    es = np.zeros(len(valid_idx))
    if not np.any(valid_mask):
        return None  # skip cluster if all hits are "Skip"

    #Filling the array
    for sysid in np.unique(ids[valid_mask]):
        if sysid not in system2name:
            #print(f"Unknown id: {sysid}, from cluster {j}, from event {i}, with energy {cluster_energy[i][j]}")
            continue
        sysname = system2name[sysid]
        mask = valid_ids == sysid
        idxs = valid_idx[mask]
        #mask = (ids == sysid) & valid_mask
        #idxs = indices[mask]
        xs[mask] = pos[sysname]["x"][i][idxs]
        ys[mask] = pos[sysname]["y"][i][idxs]
        zs[mask] = pos[sysname]["z"][i][idxs]
        ts[mask] = time[sysname][i][idxs]
        es[mask] = ener[sysname][i][idxs]
    
    # Making finite
    finite_mask = ( #Protects against naans 
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
    inertia_tensor = np.einsum('ni,nj,n->ij', diffs, diffs, ws) / W #This is the same as the equation from the paper
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    axis = eigvecs[:, -1] #extracts the eigenvector corresponding to the largest eigenvalue eigvals[-1]
    if np.dot(axis, cog) < 0:
        axis = -axis
    sigma_L2 = eigvals[-1] #longitudinal second moment #is the eigenvalue corresponding to the longitudinal axis
    sigma_T2 = eigvals[0] + eigvals[1] #variance along two orthonormal directions
    sigma_Lsq = np.sqrt(eigvals[-1])
    sigma_Tsq = np.sqrt(eigvals[0] + eigvals[1])

    #RMS opening angle radians
    theta_rms = np.arctan2(sigma_Tsq, sigma_Lsq)

    #Momentum ratios
    #moment ratio = shape profile, typical of hadrons (used in fast selection)
    #aspect ratio = width / length (intuitive geometrically)
    #theta opening, measures how fast the shower expands outward from its central direction of motion
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
    #getting the two highest indice cell
    top2_idx = np.argsort(ws)[-2:]

    #lateral energy dispersion, set r_i = 0 for top 2 energeticcells
    r_out = radials.copy()
    r_out[top2_idx] = 0.0
    r2_out = np.sum(ws* (r_out**2)) / W

    #r_core
    r_core_arr = np.zeros_like(radials)
    r_core_arr[top2_idx] = rcore
    r2_core = np.sum(ws * (r_core_arr**2)) / W

    m2_lat = r2_out / (r2_out + r2_core)

    #Longitudinal dispursion
    l_out = lambdas.copy()
    l_out[top2_idx] = 0.0
    l2_out = np.sum(ws*(l_out**2)) / W

    #l_core
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
    # Get the range for track states belonging to this track
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
    #Summing over Z^kappa theta ^ alpha
    #theta = delta Ri / R
    #Zi = energy_i / energy_T
    #This will be from the interaction point, could also be done from the reference point
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
    # Final-state MC particles
    mask = (status_mc[i] == 1)
    mx = np.asarray(mc_px[i])[mask]
    my = np.asarray(mc_py[i])[mask]
    mz = np.asarray(mc_pz[i])[mask]
    if len(mx) == 0:
        return None

    mc_r = np.sqrt(mx**2 + my**2 + mz**2)
    mc_theta = np.arccos(mz / mc_r)
    mc_phi = np.arctan2(my, mx)

    # Cluster centers and energies
    num_clusters = len(hits_begin_all[i])
    clus_ids = []   # original cluster indices (some may be skipped)
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

    # Shape (n_clusters, n_mc): every cluster against every MC particle
    cosang = (np.sin(c_theta)[:, None] * np.sin(mc_theta)[None, :]
              * np.cos(c_phi[:, None] - mc_phi[None, :])
              + np.cos(c_theta)[:, None] * np.cos(mc_theta)[None, :])
    cosang = np.clip(cosang, -1.0, 1.0)
    angular_distance = np.arccos(cosang)

    # Candidate if within bound of any MC particle
    is_candidate = (angular_distance <= bound).any(axis=1)
    if not is_candidate.any():
        return None

    # Highest-energy candidate wins
    cand = np.nonzero(is_candidate)[0]
    best = cand[np.argmax(energy[cand])]
    return int(clus_ids[best]), 

def cluster_response(event, cluster_energy):
    """(E_cluster - E_true) / E_true, as a scalar. E_true sums final-state MC particles."""
    if cluster_energy is None:
        return np.nan
    mask = np.asarray(status_mc[event]) == 1
    px = np.asarray(mc_px[event], dtype=float)[mask]
    py = np.asarray(mc_py[event], dtype=float)[mask]
    pz = np.asarray(mc_pz[event], dtype=float)[mask]
    m = np.asarray(mc_mass[event], dtype=float)[mask]
    mc_energy = np.sum(np.sqrt(m**2 + px**2 + py**2 + pz**2))
    if mc_energy <= 0:
        return np.nan, np.nan
    return cluster_energy / mc_energy, mc_energy


all_particle_data = {}
all_time_data = {}
all_isolation_data = {}
all_matched_iso = {}
for pdg, particle_info in particles.items():
    file = uproot.open(particle_info["file"])
    events = file["events"]
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
    pandora_tracks_end   = events["PandoraPFOs/PandoraPFOs.tracks_end"].array()
    track_states_x = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.x"].array()
    track_states_y = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.y"].array()
    track_states_z = events["_AllTracks_trackStates/_AllTracks_trackStates.referencePoint.z"].array()
    pandora_track_statesbegin = events["AllTracks/AllTracks.trackStates_begin"].array()
    pandora_track_statesend = events["AllTracks/AllTracks.trackStates_end"].array()
    pandora_tracks_index = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"].array()
    pos, ener, time, cellid, location = load_calorimeter_arrays()
    pfo_energy = events["PandoraPFOs/PandoraPFOs.energy"].array()
    pfo_pid = events["PandoraPFOs/PandoraPFOs.PDG"].array()
    track_states_location = events["_AllTracks_trackStates/_AllTracks_trackStates.location"].array()
    mcparticles = events["MCParticles"]
    status_mc = mcparticles["MCParticles.generatorStatus"].array()
    mc_x = mcparticles["MCParticles.endpoint.x"].array()
    mc_y = mcparticles["MCParticles.endpoint.y"].array()
    mc_z = mcparticles["MCParticles.endpoint.z"].array()
    mc_px = mcparticles["MCParticles.momentum.x"].array()
    mc_py = mcparticles["MCParticles.momentum.y"].array()
    mc_pz = mcparticles["MCParticles.momentum.z"].array()
    mc_mass = mcparticles["MCParticles.mass"].array()

    cluster_data = {
        "barycenters_eta": [],
        "barycenters_phi": [],
        "cog_x": [],
        "cog_y": [],
        "cog_z": [],
        "long_first": [],
        "radial_first": [],
        "longcalo_first": [],
        "semi_length_L2": [],
        "semi_length_T2": [],
        "width_measure": [],
        "length_measure": [],
        "effective_eta_phi": [],
        "zeroth_moment": [],
        "aspect_ratios": [],
        "theta_rms_all": [],
        "radial_rms_all": [],
        "longitudinal_rms_all": [],
        "r_thirdmoment": [],
        "r_fourthmoment": [],
        "r_skewness": [],
        "r_kurtosis": [],
        "l_thirdmoment": [],
        "l_fourthmoment": [],
        "l_skewness": [],
        "l_kurtosis": [],
        "response": [],
    }

    cluster_data_time = {
        "time_energyweighted_center": [],
        "time_center": [],
        "time_moment2_variance": [],
        "time_moment2_variance_sr": [],
        "time_moment3_asymm": [],
        "time_moment4_tails": [],
        "response": [],

    }
    cluster_data_signalcomp = {
        "EM_fraction": [],
        "HAD_fraction": [],
        "most_energetic_hit_all": [],
        "most_energetic_hit_em": [],
        "most_energetic_hit_had": [],
        "core_signal_frac_all": [],
        "response": [],
    }
    isolation_alone = {
        "isolation_all": [],
        "isolation_match": [],
        "isolation_hits": [],
        "isolation_angular": [],
        "isolation_hits_angular": [],
    }

    cluster_data_angularity = {
        "z_1_alpha_0_5": [],
        "z_2_alpha_1": [],
        "z_2_alpha_2": [],
        "z_3_alpha_1": [],
        "z_3_alpha_2": [],
        "response": [],
    }
    cluster_data_extra = {
        "core_signal": [],
        "hcal_ratio": [],
        "hcal_over_all": [],
        "ecal_over_all": [],
        "hcal_over_hcal": [],
        "ecal_over_ecal": [],
        "response": [],
        "mc_energy": []
    }
    isolation_matching = {
        "isolation_hits_but_all": [],
        "isolation_hits_but_clus": [],
    }
    some_width = particle_info["test_width"]
    cluster_list = []
    cluster_list2 = []
    least_isolated = []
    most_isolated = []
    mean_isolated = []
    least_isolated_hits = []
    most_isolated_hits = []
    mean_isolated_hits = []
    mean_isolated_angular = []
    least_isolated_angular = []
    most_isolated_angular = []

    #Now I want to do it with just the "matched clusters"
    cluster_axis_length = []
    cluster_clus_iso = []
    cluster_all_iso = []
    cluster_energy_list = []
    cluster_iso_fraction = []
      
    for index in range(events.num_entries):
        unclaimed = unclaimed_indexes(index, len(cluster_energy[index]))
        if len(pfo_energy[index]) == 1:
            for pfo in range(len(pfo_energy[index])):
                cluster_begin = pandora_clusters_begin[index][pfo]
                cluster_end = pandora_clusters_end[index][pfo]
                tracks_begin = pandora_tracks_begin[index][pfo]
                tracks_end = pandora_tracks_end[index][pfo]
                entry_point = None
                if tracks_begin != tracks_end:
                    for track in range(tracks_begin, tracks_end):
                        track_index = pandora_tracks_index[index][track]
                        entry = get_entry_point_from_track(index, track_index)
                        if entry is not None:
                            entry_point = list(entry)
                for cluster in range(cluster_begin, cluster_end):
                    cluster_index = pandora_clusters_index[index][cluster]
                    total_energy = zeroth_moment(index, cluster_index)
                    response, mc_energy = cluster_response(index, total_energy)
                    #cluster_data_extra["mc_energy"].append(mc_energy)
                    eta_bary, phi_bary, [cog_x, cog_y, cog_z], xs, ys, zs, ws, id_cell, locs = return_barycenter_centre_of_gravity(index, cluster_index)
                    core_signal_frac = core_signal_fraction(ws, id_cell, locs)
                    core_signal_frac2, hcalratio, hcal_over_all, ecal_over_all, hcal_over_hcal, ecal_over_ecal = core_signal_fraction_detector(ws, id_cell, locs)
                    cluster_data_extra["core_signal"].append(core_signal_frac2)
                    cluster_data_extra["hcal_ratio"].append(hcalratio)
                    cluster_data_extra["hcal_over_all"].append(hcal_over_all)
                    cluster_data_extra["ecal_over_all"].append(ecal_over_all)
                    cluster_data_extra["ecal_over_ecal"].append(ecal_over_ecal)
                    cluster_data_extra["hcal_over_hcal"].append(hcal_over_hcal)
                    cluster_data_extra["response"].append(response)
                    axis, sigma_L2, sigma_T2, sigma_L2sq, sigma_T2sq, theta_rms, _moment_ratio, aspect_ratio= get_shower_axis(xs, ys, zs, ws, [cog_x, cog_y, cog_z])
                    iso_all, iso_clus = isolation_allhits_but(
                        unclaimed, cluster_index, index, max_R=(sigma_T2sq <= 12))
                    if iso_all is None or iso_clus is None:
                        iso_all, iso_clus, iso_frac = np.nan, np.nan, np.nan
                    elif iso_all == 0:
                        iso_frac = -1
                    else:
                        iso_frac = iso_clus / iso_all
                    cluster_all_iso.append(iso_all)
                    cluster_clus_iso.append(iso_clus)
                    cluster_iso_fraction.append(iso_frac)
                    cluster_data_extra["mc_energy"].append(mc_energy)
                    rms_radial, rms_longitudinal, r_firstmoment, l_firstmoment, l_firstmoment_entry = get_rms_cluster_dimensions(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis, entry_point)
                    m2_lat, m2_long = get_normalized_energy_disp(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis)
                    sigma_eta_phi = get_effective_cluster_size(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis, eta_bary, phi_bary)
                    r_thirdmoment, l_thirdmoment, r_fourthmoment, l_fourthmoment, r_skewness, l_skewness, r_kurtosis, l_kurtosis = get_third_fourth_moments(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis)
                    time_center, energy_weighted_center, ts, es = return_time_median_enertime_median(index, cluster_index)
                    time_moment2, time_moment2_sr = return_time_moments(ts, es, energy_weighted_center, 2)
                    time_moment3, time_moment3_sr = return_time_moments(ts, es, energy_weighted_center, 3)
                    time_moment4, time_moment4_sr = return_time_moments(ts, es, energy_weighted_center, 4)
                    k_1_a_0_5 = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa = 1.0, alpha = 0.5, R = 1.0)
                    k_2_a_1 = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa = 2.0, alpha = 1.0, R = 1.0)
                    k_2_a_2 = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa = 2.0, alpha = 2.0, R = 1.0)
                    k_3_a_1 = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa = 3.0, alpha = 1.0, R = 1.0)
                    k_3_a_2 = jet_angularity(xs, ys, zs, ws, eta_bary, phi_bary, axis, kappa = 3.0, alpha = 2.0, R = 1.0)
                    cluster_data_time["time_energyweighted_center"].append(energy_weighted_center)
                    cluster_data_time["time_center"].append(time_center)
                    cluster_data_time["time_moment2_variance"].append(time_moment2)
                    cluster_data_time["time_moment2_variance_sr"].append(time_moment2_sr)
                    cluster_data_time["time_moment3_asymm"].append(time_moment3)
                    cluster_data_time["time_moment4_tails"].append(time_moment4)
                    cluster_data_time["response"].append(response)
                    cluster_data["response"].append(response)
                    cluster_data["zeroth_moment"].append(total_energy)
                    cluster_data["barycenters_eta"].append(eta_bary)
                    cluster_data["barycenters_phi"].append(phi_bary)
                    cluster_data["cog_x"].append(cog_x)
                    cluster_data["cog_y"].append(cog_y)
                    cluster_data["cog_z"].append(cog_z)
                    cluster_data["radial_first"].append(r_firstmoment)
                    cluster_data["long_first"].append(l_firstmoment)
                    cluster_data["longcalo_first"].append(np.nan if l_firstmoment_entry is None else l_firstmoment_entry)
                    cluster_data["semi_length_L2"].append(sigma_L2)
                    cluster_data["semi_length_T2"].append(sigma_T2)
                    cluster_data["radial_rms_all"].append(rms_radial)
                    cluster_data["longitudinal_rms_all"].append(rms_longitudinal)
                    cluster_data["theta_rms_all"].append(theta_rms)
                    cluster_data["aspect_ratios"].append(aspect_ratio)
                    cluster_data["effective_eta_phi"].append(sigma_eta_phi)
                    cluster_data["width_measure"].append(m2_lat)
                    cluster_data["length_measure"].append(m2_long)
                    cluster_data["r_thirdmoment"].append(r_thirdmoment)
                    cluster_data["r_fourthmoment"].append(r_fourthmoment)
                    cluster_data["r_skewness"].append(r_skewness)
                    cluster_data["r_kurtosis"].append(r_kurtosis)
                    cluster_data["l_thirdmoment"].append(l_thirdmoment)
                    cluster_data["l_fourthmoment"].append(l_fourthmoment)
                    cluster_data["l_skewness"].append(l_skewness)
                    cluster_data["l_kurtosis"].append(l_kurtosis)

                    cluster_data_angularity["z_1_alpha_0_5"].append(k_1_a_0_5)
                    cluster_data_angularity["z_2_alpha_1"].append(k_2_a_1)
                    cluster_data_angularity["z_2_alpha_2"].append(k_2_a_2)
                    cluster_data_angularity["z_3_alpha_1"].append(k_3_a_1)
                    cluster_data_angularity["z_3_alpha_2"].append(k_3_a_2)
                    cluster_data_angularity["response"].append(response)

                    cluster_data_signalcomp["core_signal_frac_all"].append(core_signal_frac)
                    cluster_data_signalcomp["response"].append(response)


    #Let's have a plotting function
    #Let's just make a basic rudimentary plot for all of these, bar plot counts vs whatever variable this is
    #plot_cluster_variable(cluster_data, bins=50, save_prefix="moment_spatial", particle_name=particle_info["name"], color=particle_info["color"])

    #Let's have a plotting function
    #Let's just make a basic rudimentary plot for all of these, bar plot counts vs whatever variable this is
    #plot_cluster_variable(cluster_data, bins=50, save_prefix="moment_spatial_time", particle_name=particle_info["name"], color=particle_info["color"])
    for index in range(events.num_entries):
        if len(pfo_energy[index]) == 1:
            for pfo in range(len(pfo_energy[index])):
                cluster_begin = pandora_clusters_begin[index][pfo]
                cluster_end = pandora_clusters_end[index][pfo]
                tracks_begin = pandora_tracks_begin[index][pfo]
                tracks_end = pandora_tracks_end[index][pfo]
                entry_point = None
                if (tracks_begin != tracks_end):
                    for track in range (tracks_begin, tracks_end): #this is most nearly definitely only 1 track
                        track_index = pandora_tracks_index[index][track]
                        x_entry, y_entry, z_entry = get_entry_point_from_track(index, track_index)
                        entry_point = [x_entry, y_entry, z_entry]
                for cluster in range(cluster_begin, cluster_end):
                    cluster_index = pandora_clusters_index[index][cluster]
                    em_total, had_total = return_EM_total_HAD_total(index, cluster_index)
                    total_energy = zeroth_moment(index, cluster_index)
                    max_hit, max_hit_em, max_hit_had = return_max_hit(index, cluster_index)
                    em_frac = em_total / total_energy
                    had_frac = had_total / total_energy
                    max_frac = max_hit / total_energy 
                    max_frac_em = max_hit_em / total_energy
                    max_frac_had = max_hit_had / total_energy
                    cluster_data_signalcomp["EM_fraction"].append(em_frac)
                    cluster_data_signalcomp["HAD_fraction"].append(had_frac)
                    cluster_data_signalcomp["most_energetic_hit_all"].append(max_frac)
                    cluster_data_signalcomp["most_energetic_hit_em"].append(max_frac_em)
                    cluster_data_signalcomp["most_energetic_hit_had"].append(max_frac_had)

    # Build the feature matrix for this particle

    columns = {
        "clus_iso": cluster_clus_iso,
        "all_iso": cluster_all_iso,
        "iso_fraction": cluster_iso_fraction,
        "cog_z": cluster_data["cog_z"],
        "radial_first": cluster_data["radial_first"],
        "longcalo_first": cluster_data["longcalo_first"],
        "aspect_ratio": cluster_data["aspect_ratios"],
        "radial_rms": cluster_data["radial_rms_all"],
        "time_center": cluster_data_time["time_center"],
        "HAD_fraction": cluster_data_signalcomp["HAD_fraction"],
        "max_hit_had": cluster_data_signalcomp["most_energetic_hit_had"],
        "core_signal_frac": cluster_data_signalcomp["core_signal_frac_all"],
        "hcal_over_all": cluster_data_extra["hcal_over_all"],
        "ecal_over_all": cluster_data_extra["ecal_over_all"],
        "z_3_alpha_1": cluster_data_angularity["z_3_alpha_1"],
    }

    # Sanity check: every feature must have the same length

    lengths = {k: len(v) for k, v in columns.items()}

    print(particle_info["name"], lengths)

    assert len(set(lengths.values())) == 1, "Feature lengths don't match!"

    X_particle = np.column_stack([
        np.asarray(v, dtype=float) for v in columns.values()
    ])

    # Drop rows with NaN/inf

    good = np.isfinite(X_particle).all(axis=1)

    print(f"Keeping {good.sum()} / {len(good)} {particle_info['name']} clusters")

    X_particle = X_particle[good]

    # Electron = 0, pion = 1

    label = 0 if pdg == 11 else 1

    y_particle = np.full(len(X_particle), label, dtype=int)

    # Save this particle's data so we can combine them later

    all_particle_data[pdg] = (X_particle, y_particle)

    #cluster_clus_iso
    #cluster_all_iso
    #cluster_iso_fraction

# Sanity check: every feature must have the same length


######################### Combine particles #########################

# Combine electron and pion feature matrices

X = np.vstack([
    all_particle_data[11][0],
    all_particle_data[211][0]
])

y = np.concatenate([
    all_particle_data[11][1],
    all_particle_data[211][1]
])

print("Total electrons:", np.sum(y == 0))
print("Total pions:", np.sum(y == 1))

######################### Train / test #########################

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance the training/development dataset

rng = np.random.default_rng(42)

electron_idx = np.where(y_dev == 0)[0]
pion_idx = np.where(y_dev == 1)[0]

n_each = min(len(electron_idx), len(pion_idx))

electron_idx = rng.choice(electron_idx, n_each, replace=False)
pion_idx = rng.choice(pion_idx, n_each, replace=False)

balanced_idx = np.concatenate([electron_idx, pion_idx])
rng.shuffle(balanced_idx)

X_dev = X_dev[balanced_idx]
y_dev = y_dev[balanced_idx]

print("Balanced training electrons:", np.sum(y_dev == 0))
print("Balanced training pions:", np.sum(y_dev == 1))

# K-fold validation

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

histories = []

models, scalers, validation_scores = [], [], []

n_epochs = 150

for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev, y_dev)):

    X_train, X_val = X_dev[train_idx], X_dev[val_idx]
    y_train, y_val = y_dev[train_idx], y_dev[val_idx]

    scaler = StandardScaler().fit(X_train)

    Xtr = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-2,
        random_state=42
    )

    train_loss, val_loss = [], []

    for epoch in range(n_epochs):

        if epoch == 0:
            mlp.partial_fit(Xtr, y_train, classes=[0, 1])
        else:
            mlp.partial_fit(Xtr, y_train)

        train_loss.append(
            log_loss(y_train, mlp.predict_proba(Xtr), labels=[0, 1])
        )

        val_loss.append(
            log_loss(y_val, mlp.predict_proba(Xv), labels=[0, 1])
        )

        if epoch % 10 == 0:
            print(
                f"fold {fold} epoch {epoch:4d}  "
                f"train loss {train_loss[-1]:.5f}  "
                f"val loss {val_loss[-1]:.5f}",
                flush=True
            )

    histories.append((train_loss, val_loss))

    models.append(mlp)
    scalers.append(scaler)

    val_predictions = mlp.predict_proba(Xv)[:, 1]

    validation_scores.append(
        roc_auc_score(y_val, val_predictions)
    )


######################### Checking Results #########################

print("Validation AUC:", np.round(validation_scores, 4))
print(f"Mean validation AUC: {np.mean(validation_scores):.4f}")

# Ensemble the fold models on the held-out test set

predictions_prob = np.mean(
    [
        m.predict_proba(s.transform(X_test))[:, 1]
        for m, s in zip(models, scalers)
    ],
    axis=0
)

predictions = (predictions_prob >= 0.5).astype(int)

print("Test accuracy:", accuracy_score(y_test, predictions))
print("Test AUC:", roc_auc_score(y_test, predictions_prob))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, predictions_prob)
test_auc = roc_auc_score(y_test, predictions_prob)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(fpr, tpr, label=f"MLP (AUC = {test_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", label="Random classifier")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Electron vs Pion Classification")
ax.legend()
ax.grid(True, alpha=0.3)

fig.savefig("mlp_roc_curve.png", dpi=150, bbox_inches="tight")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Electron", "Pion"]
))


######################### Training Losses #########################





fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, (tl, vl), k in zip(axes, histories, range(5)):
    ax.plot(tl, label="train")
    ax.plot(vl, label="val")
    ax.set_title(f"fold {k}")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
axes[0].set_ylabel("MSE")
axes[0].legend()
fig.savefig("mlp_train_val_curves.png", dpi=150, bbox_inches="tight")

lo = min(y_test.min(), predictions.min())
hi = max(y_test.max(), predictions.max())
bins = np.linspace(lo, hi, 40)   # one set of bins used for both

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(y_test, bins=bins, histtype="stepfilled", alpha=0.45,
        color="royalblue",
        label=f"actual (mean={y_test.mean():.3f}, σ={y_test.std():.3f})")
ax.hist(predictions, bins=bins, histtype="stepfilled", alpha=0.45,
        color="darkorange",
        label=f"predicted (mean={predictions.mean():.3f}, σ={predictions.std():.3f})")
ax.hist(y_test, bins=bins, histtype="step", color="royalblue", lw=1.5)
ax.hist(predictions, bins=bins, histtype="step", color="darkorange", lw=1.5)
ax.set_xlabel("Labels ( 0 = electron, 1 = pion) ")
ax.set_title("Test set: actual vs predicted response")
ax.legend()
fig.savefig("mlp_actual_vs_pred_hist.png", dpi=150, bbox_inches="tight")