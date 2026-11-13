#Barycenter (eta and phi)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
choices = [50]
import awkward as ak
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
import corner
from matplotlib.backends.backend_pdf import PdfPages
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

'''
particles = {
    11: {
        "name": "electron",
        "color": "green",
        "file": "/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_11_nobib0.edm4hep.root",
    },
    211: {
        "name": "pion",
        "color": "royalblue",
        "file": "/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root",
    },
    
    2112: {
        "name": "neutron",
        "color": "red",
        "file": "/users/rldohert/data/mucoll/rldohert/pdg_2112_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_2112_nobib0.edm4hep.root",
    }, 
}
'''
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
    
def zeroth_moment_time(i, j): #event, cluster
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
    ts = np.zeros(len(valid_idx))
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
        ts[mask] = time[sysname][i][idxs]
    
    finite_mask = ( #Protects against naans 
        np.isfinite(ts)
    )
    ts = ts[finite_mask]
    T_tot = np.sum(ts)
    return T_tot
    
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


def return_barycenter_centre_of_time(i, j): #event, cluster
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
    
    # Making finite
    finite_mask = ( #Protects against naans 
        np.isfinite(xs)
        & np.isfinite(ys)
        & np.isfinite(zs)
        & np.isfinite(ts)
    )
    xs = xs[finite_mask]
    ys = ys[finite_mask]
    zs = zs[finite_mask]
    ts = ts[finite_mask]

    r3 = np.sqrt(xs**2 + ys**2 + zs**2)
    rho = np.sqrt(xs**2 + ys**2)
    phi = np.arctan2(ys, xs)

    theta = np.arctan2(rho, zs)
    eta = -np.log(np.tan(theta / 2.0))

    T_tot = np.sum(ts)
    if T_tot == 0:
        print(f"Warning: zero total energy in event {i}, cluster {j}")
        return None
    eta_bary = np.sum(ts * eta) / T_tot
    phi_bary = np.arctan2(np.sum(ts * np.sin(phi)), np.sum(ts * np.cos(phi)))

    cog_x = np.sum(ts * xs) / T_tot
    cog_y = np.sum(ts * ys) / T_tot
    cog_z = np.sum(ts * zs) / T_tot
    return eta_bary, phi_bary, [cog_x, cog_y, cog_z], xs, ys, zs, ts


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

def isolation_allhits_but(unclaimed, cluster_idx, event_index, ws, cell_ids, locs, notwidth):
    #number of hits our cluster
    
    hits_begin_arr = hits_begin_all[event_index]
    hits_end_arr = hits_end_all[event_index]
    hit_index = hit_index_all[event_index]
    collection_ID = collectionID_all[event_index]

    #This clusters indices
    start, end = hits_begin_arr[cluster_idx], hits_end_arr[cluster_idx]
    num_hits_ours = end - start
    
    own_global_idx = np.asarray(hit_index[start:end])
    own_idxs        = np.asarray(collection_ID[start:end])

    #Filling this cluster getting a dictionary filled with cell indexes
    own_idx_by_sys = {name: set() for name in real_systems}
    own_E_by_sys = {name: {} for name in real_systems}
    ws_arr = np.asarray(ws)
    for sysid, gidx, w in zip(own_idxs, own_global_idx, ws_arr):
        sysname = system2name.get(sysid, None)
        if sysname is None or sysname == "Skip":
            continue
        own_idx_by_sys[sysname].add(int(gidx))
        own_E_by_sys[sysname][int(gidx)] = w

    #For this cluster
    cell_ids_arr = np.asarray(cell_ids).astype(np.uint64)
    layer_of_hit = ((cell_ids_arr >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
    locs_arr = np.asarray(locs)
    

    layer_energy = {} #For our cluster getting the energy in each layer
    for sysname, lay, w in zip(locs_arr, layer_of_hit, ws_arr):
        key = (sysname, int(lay))
        layer_energy[key] = layer_energy.get(key, 0.0) + w
    #gets an array for detector location filling by layer the energy for our cluster

    numerator_all = 0.0
    denominator_all = 0.0
    numerator_clus = 0.0
    denominator_clus = 0.0

    #Fill the position for this event
    for (locname, lay), E_s in layer_energy.items(): #going through each layer for our known cluster
        if E_s <= 0:
            continue
        sysname = loc2sys.get(locname)
        if sysname is None or sysname not in real_systems:
            continue
        
        full_cellid = np.asarray(cellid[sysname][event_index]).astype(np.uint64)
        full_layer = ((full_cellid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
        #["x"][i][idxs]
        pos_x = np.asarray(pos[sysname]["x"][event_index])
        pos_y = np.asarray(pos[sysname]["y"][event_index])
        pos_z = np.asarray(pos[sysname]["z"][event_index])
        layer_mask = full_layer == lay
        layer_global_idx = np.nonzero(layer_mask)[0]
        #use layer_mask, should work on the hits since they're made together

        #exclude our clusters hits

        our_idx = np.array([gi for gi in layer_global_idx if gi in own_idx_by_sys[sysname]], dtype=int)
        candidate_idx = np.array([gi for gi in layer_global_idx if gi not in own_idx_by_sys[sysname]], dtype=int)

        if len(candidate_idx) == 0:
            continue
        w = np.array([own_E_by_sys[sysname][int(gi)] for gi in our_idx])
        pos_x_ourclus = pos_x[our_idx]
        pos_y_ourclus = pos_y[our_idx]
        pos_z_ourclus = pos_z[our_idx]
        W = w.sum()
        #Let's do energy weighted centroid now
        cx = np.sum(w * pos_x_ourclus) / W
        cy = np.sum(w * pos_y_ourclus) / W
        cz = np.sum(w * pos_z_ourclus) / W

        #Now energy weighted width
        d2_own = (pos_x_ourclus - cx)**2 + (pos_y_ourclus - cy)**2 + (pos_z_ourclus - cz) **2
        width = np.sqrt(np.sum(w * d2_own) / W)

        some_width = width / 2
        #I want the energy per hits too 

        pos_x_notclus = pos_x[candidate_idx]
        pos_y_notclus = pos_y[candidate_idx]
        pos_z_notclus = pos_z[candidate_idx]
        unclaimed_set = set(unclaimed[sysname]) #just turns into a set

        full_collection = np.zeros(len(pos_x_notclus))
        for index in range(len(pos_x_ourclus)):
            pos_x_ours = pos_x_ourclus[index]
            pos_y_ours = pos_y_ourclus[index]
            pos_z_ours = pos_z_ourclus[index]

            #Will this work without looping (delta x will be an array)
            delta_xs = pos_x_ours - pos_x_notclus
            delta_ys = pos_y_ours - pos_y_notclus
            delta_zs = pos_z_ours - pos_z_notclus

            rs = np.sqrt((delta_xs)**2 + (delta_ys)**2 + (delta_zs)**2)
            full_collection[rs < some_width] = 1
        
        full_collection = full_collection.astype(bool)
        nearby_idx = candidate_idx[full_collection]
        n_neighbor = len(nearby_idx)
        n_noclus = sum(1 for gi in nearby_idx if int(gi) in unclaimed_set)
        

        #Now we are going to find numerator and denominator
        numerator_all += E_s * (n_neighbor)
        denominator_all += E_s

        numerator_clus += E_s * (n_neighbor - n_noclus)
        denominator_clus += E_s

    total_allnearby = numerator_all / (denominator_all) if denominator_all > 0 else None
    total_clus = numerator_clus / (denominator_clus) if denominator_clus > 0 else None
    return total_allnearby, total_clus



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


def plot_cluster_variable(data_dict, bins=50, save_prefix="cluster_plot", particle_name="", color="green"):
    for var_name, values in data_dict.items():
        clean_vals = [v for v in values if v is not None and not np.isnan(v)]
        if len(clean_vals) == 0:
            print(f"Skipping {var_name}: No valid data points.")
            continue
        plt.figure(figsize=(7, 5))
        plt.hist(clean_vals, bins=bins, color=color, edgecolor='black', alpha=0.75)
        plt.title(f"Distribution of {var_name} - {particle_name}", fontsize=12)
        plt.xlabel(var_name, fontsize=10)
        plt.ylabel("Counts", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{save_prefix}_{var_name}_{particle_name}.png",dpi=150)
        plt.close()
    print(f"Finished Generating the plots")


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

#We are going to do isolation_allhits_etaphi

def isolation_allhits_etaphi(cluster_idx, event_index, ws, cell_ids, locs, angular_width, eta_center, phi_center):
    #number of hits our cluster
    
    hits_begin_arr = hits_begin_all[event_index]
    hits_end_arr = hits_end_all[event_index]
    hit_index = hit_index_all[event_index]
    collection_ID = collectionID_all[event_index]

    #This clusters indices
    start, end = hits_begin_arr[cluster_idx], hits_end_arr[cluster_idx]
    num_hits_ours = end - start
    
    own_global_idx = np.asarray(hit_index[start:end])
    own_idxs        = np.asarray(collection_ID[start:end])

    #Filling this cluster getting a dictionary filled with cell indexes
    own_idx_by_sys = {name: set() for name in real_systems}
    for sysid, gidx in zip(own_idxs, own_global_idx):
        sysname = system2name.get(sysid, None)
        if sysname is None or sysname == "Skip":
            continue
        own_idx_by_sys[sysname].add(int(gidx))

    #For this cluster
    cell_ids_arr = np.asarray(cell_ids).astype(np.uint64)
    layer_of_hit = ((cell_ids_arr >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
    locs_arr = np.asarray(locs)
    ws_arr = np.asarray(ws)

    layer_energy = {} #For our cluster getting the energy in each layer
    for sysname, lay, w in zip(locs_arr, layer_of_hit, ws_arr):
        key = (sysname, int(lay))
        layer_energy[key] = layer_energy.get(key, 0.0) + w
    #gets an array for detector location filling by layer the energy for our cluster

    numerator = 0.0
    denominator = 0.0

    #Fill the position for this event
    for (locname, lay), E_s in layer_energy.items(): #going through each layer for our known cluster
        if E_s <= 0:
            continue
        sysname = loc2sys.get(locname)
        if sysname is None or sysname not in real_systems:
            continue
        
        full_cellid = np.asarray(cellid[sysname][event_index]).astype(np.uint64)
        full_layer = ((full_cellid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
        #["x"][i][idxs]
        pos_x = np.asarray(pos[sysname]["x"][event_index])
        pos_y = np.asarray(pos[sysname]["y"][event_index])
        pos_z = np.asarray(pos[sysname]["z"][event_index])
        layer_mask = full_layer == lay
        layer_global_idx = np.nonzero(layer_mask)[0]
        #use layer_mask, should work on the hits since they're made together

        #exclude our clusters hits

        our_idx = np.array([gi for gi in layer_global_idx if gi in own_idx_by_sys[sysname]], dtype=int)
        candidate_idx = np.array([gi for gi in layer_global_idx if gi not in own_idx_by_sys[sysname]], dtype=int)

        if len(candidate_idx) == 0:
            continue
        
        pos_x_ourclus = pos_x[our_idx]
        pos_y_ourclus = pos_y[our_idx]
        pos_z_ourclus = pos_z[our_idx]

        pos_x_notclus = pos_x[candidate_idx]
        pos_y_notclus = pos_y[candidate_idx]
        pos_z_notclus = pos_z[candidate_idx]

        full_collection = np.zeros(len(pos_x_notclus))
        phi_notclus = np.arctan2(pos_y_notclus, pos_x_notclus)
        r_notclus = np.sqrt(pos_x_notclus**2 + pos_y_notclus**2)
        theta_notclus = np.arctan2(r_notclus, pos_z_notclus)
        eta_notclus = -np.log(np.tan(theta_notclus / 2))

        delta_eta_not = eta_center - eta_notclus
        dphi_not = phi_center - phi_notclus

        delta_phi_not = np.arctan2(np.sin(dphi_not), np.cos(dphi_not))
        r_not = np.sqrt(delta_phi_not**2 + delta_eta_not**2)
        full_collection[r_not < angular_width] = 1
        full_collection = full_collection.astype(bool)
        nearby_idx = candidate_idx[full_collection]
        n_neighbor = len(nearby_idx)
        numerator += E_s * (n_neighbor)
        denominator += E_s
    total = numerator / (denominator * num_hits_ours) if denominator > 0 else None
    return total


def isolation_allhits(cluster_idx, event_index, ws, cell_ids, locs, some_width):
    #number of hits our cluster
    
    hits_begin_arr = hits_begin_all[event_index]
    hits_end_arr = hits_end_all[event_index]
    hit_index = hit_index_all[event_index]
    collection_ID = collectionID_all[event_index]

    #This clusters indices
    start, end = hits_begin_arr[cluster_idx], hits_end_arr[cluster_idx]
    num_hits_ours = end - start
    
    own_global_idx = np.asarray(hit_index[start:end])
    own_idxs        = np.asarray(collection_ID[start:end])

    #Filling this cluster getting a dictionary filled with cell indexes
    own_idx_by_sys = {name: set() for name in real_systems}
    for sysid, gidx in zip(own_idxs, own_global_idx):
        sysname = system2name.get(sysid, None)
        if sysname is None or sysname == "Skip":
            continue
        own_idx_by_sys[sysname].add(int(gidx))

    #For this cluster
    cell_ids_arr = np.asarray(cell_ids).astype(np.uint64)
    layer_of_hit = ((cell_ids_arr >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
    locs_arr = np.asarray(locs)
    ws_arr = np.asarray(ws)

    layer_energy = {} #For our cluster getting the energy in each layer
    for sysname, lay, w in zip(locs_arr, layer_of_hit, ws_arr):
        key = (sysname, int(lay))
        layer_energy[key] = layer_energy.get(key, 0.0) + w
    #gets an array for detector location filling by layer the energy for our cluster

    numerator = 0.0
    denominator = 0.0

    #Fill the position for this event
    for (locname, lay), E_s in layer_energy.items(): #going through each layer for our known cluster
        if E_s <= 0:
            continue
        sysname = loc2sys.get(locname)
        if sysname is None or sysname not in real_systems:
            continue
        
        full_cellid = np.asarray(cellid[sysname][event_index]).astype(np.uint64)
        full_layer = ((full_cellid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
        #["x"][i][idxs]
        pos_x = np.asarray(pos[sysname]["x"][event_index])
        pos_y = np.asarray(pos[sysname]["y"][event_index])
        pos_z = np.asarray(pos[sysname]["z"][event_index])
        layer_mask = full_layer == lay
        layer_global_idx = np.nonzero(layer_mask)[0]
        #use layer_mask, should work on the hits since they're made together

        #exclude our clusters hits

        our_idx = np.array([gi for gi in layer_global_idx if gi in own_idx_by_sys[sysname]], dtype=int)
        candidate_idx = np.array([gi for gi in layer_global_idx if gi not in own_idx_by_sys[sysname]], dtype=int)

        if len(candidate_idx) == 0:
            continue
        
        pos_x_ourclus = pos_x[our_idx]
        pos_y_ourclus = pos_y[our_idx]
        pos_z_ourclus = pos_z[our_idx]

        pos_x_notclus = pos_x[candidate_idx]
        pos_y_notclus = pos_y[candidate_idx]
        pos_z_notclus = pos_z[candidate_idx]

        full_collection = np.zeros(len(pos_x_notclus))
        for index in range(len(pos_x_ourclus)):
            pos_x_ours = pos_x_ourclus[index]
            pos_y_ours = pos_y_ourclus[index]
            pos_z_ours = pos_z_ourclus[index]

            #Will this work without looping (delta x will be an array)
            delta_xs = pos_x_ours - pos_x_notclus
            delta_ys = pos_y_ours - pos_y_notclus
            delta_zs = pos_z_ours - pos_z_notclus

            rs = np.sqrt((delta_xs)**2 + (delta_ys)**2 + (delta_zs)**2)
            full_collection[rs < some_width] = 1
        
        full_collection = full_collection.astype(bool)
        nearby_idx = candidate_idx[full_collection]
        n_neighbor = len(nearby_idx)
        #Now we are going to find numerator and denominator
        numerator += E_s * (n_neighbor)
        denominator += E_s
    total = numerator / (denominator * num_hits_ours) if denominator > 0 else None
    return total


def isolation(unclaimed, cluster_idx, event_index, ws, cell_ids, locs, some_width):
    #takes the cluster
    #Finds the layers every hit in the cluster go across, in each part of the detector
    #Finds the total energy from these hits
    #finds unclaimed hits in the same layer
    #totals unclaimed hits
    #also finds nearby cluster and totals those hits in the same layer (excluding unclaimed hits)
    #problem wiht cell_ids and locs only work for this cluster

    #xs, ys, zs should be the full array for this event , so the pos thing isolated for x y z 
    hits_begin_arr = hits_begin_all[event_index]
    hits_end_arr = hits_end_all[event_index]
    hit_index = hit_index_all[event_index]
    collection_ID = collectionID_all[event_index]

    #This clusters indices
    start, end = hits_begin_arr[cluster_idx], hits_end_arr[cluster_idx]
    own_global_idx = np.asarray(hit_index[start:end])
    own_idxs        = np.asarray(collection_ID[start:end])

    #Filling this cluster getting a dictionary filled with cell indexes
    own_idx_by_sys = {name: set() for name in real_systems}
    for sysid, gidx in zip(own_idxs, own_global_idx):
        sysname = system2name.get(sysid, None)
        if sysname is None or sysname == "Skip":
            continue
        own_idx_by_sys[sysname].add(int(gidx))

    #For this cluster
    cell_ids_arr = np.asarray(cell_ids).astype(np.uint64)
    layer_of_hit = ((cell_ids_arr >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
    locs_arr = np.asarray(locs)
    ws_arr = np.asarray(ws)

    layer_energy = {} #For our cluster getting the energy in each layer
    for sysname, lay, w in zip(locs_arr, layer_of_hit, ws_arr):
        key = (sysname, int(lay))
        layer_energy[key] = layer_energy.get(key, 0.0) + w
    #gets an array for detector location filling by layer the energy for our cluster

    numerator = 0.0
    denominator = 0.0

    #Fill the position for this event
    for (locname, lay), E_s in layer_energy.items(): #going through each layer for our known cluster
        if E_s <= 0:
            continue
        sysname = loc2sys.get(locname)
        if sysname is None or sysname not in real_systems:
            continue
        
        full_cellid = np.asarray(cellid[sysname][event_index]).astype(np.uint64)
        full_layer = ((full_cellid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
        #["x"][i][idxs]
        pos_x = np.asarray(pos[sysname]["x"][event_index])
        pos_y = np.asarray(pos[sysname]["y"][event_index])
        pos_z = np.asarray(pos[sysname]["z"][event_index])
        layer_mask = full_layer == lay
        layer_global_idx = np.nonzero(layer_mask)[0]
        #use layer_mask, should work on the hits since they're made together

        #exclude our clusters hits

        our_idx = np.array([gi for gi in layer_global_idx if gi in own_idx_by_sys[sysname]], dtype=int)
        candidate_idx = np.array([gi for gi in layer_global_idx if gi not in own_idx_by_sys[sysname]], dtype=int)

        if len(candidate_idx) == 0:
            continue
        
        pos_x_ourclus = pos_x[our_idx]
        pos_y_ourclus = pos_y[our_idx]
        pos_z_ourclus = pos_z[our_idx]

        pos_x_notclus = pos_x[candidate_idx]
        pos_y_notclus = pos_y[candidate_idx]
        pos_z_notclus = pos_z[candidate_idx]

        unclaimed_set = set(unclaimed[sysname]) #just turns into a set

        #Now we have to check to make sure the hits fall into the correct radius

        full_collection = np.zeros(len(pos_x_notclus))
        for index in range(len(pos_x_ourclus)):
            pos_x_ours = pos_x_ourclus[index]
            pos_y_ours = pos_y_ourclus[index]
            pos_z_ours = pos_z_ourclus[index]

            #Will this work without looping (delta x will be an array)
            delta_xs = pos_x_ours - pos_x_notclus
            delta_ys = pos_y_ours - pos_y_notclus
            delta_zs = pos_z_ours - pos_z_notclus

            rs = np.sqrt((delta_xs)**2 + (delta_ys)**2 + (delta_zs)**2)
            full_collection[rs < some_width] = 1
        
        full_collection = full_collection.astype(bool)
        nearby_idx = candidate_idx[full_collection]
        n_noclus = sum(1 for gi in nearby_idx if int(gi) in unclaimed_set)
        n_neighbor = len(nearby_idx)
        if n_neighbor == 0:
            continue

        numerator += E_s * (n_noclus / n_neighbor)
        denominator += E_s

    return numerator / denominator if denominator > 0 else None

#Let's do this via eta and phi, also weight by layer, but radially

def isolation_eta_phi(unclaimed, cluster_idx, event_index, ws, cell_ids, locs, some_width, bary_eta, bary_phi):
    #takes the cluster
    #Finds the layers every hit in the cluster go across, in each part of the detector
    #Finds the total energy from these hits
    #finds unclaimed hits in the same layer
    #totals unclaimed hits
    #also finds nearby cluster and totals those hits in the same layer (excluding unclaimed hits)
    #problem wiht cell_ids and locs only work for this cluster

    #xs, ys, zs should be the full array for this event , so the pos thing isolated for x y z 
    hits_begin_arr = hits_begin_all[event_index]
    hits_end_arr = hits_end_all[event_index]
    hit_index = hit_index_all[event_index]
    collection_ID = collectionID_all[event_index]

    #This clusters indices
    start, end = hits_begin_arr[cluster_idx], hits_end_arr[cluster_idx]
    own_global_idx = np.asarray(hit_index[start:end])
    own_idxs        = np.asarray(collection_ID[start:end])

    #Filling this cluster getting a dictionary filled with cell indexes
    own_idx_by_sys = {name: set() for name in real_systems}
    for sysid, gidx in zip(own_idxs, own_global_idx):
        sysname = system2name.get(sysid, None)
        if sysname is None or sysname == "Skip":
            continue
        own_idx_by_sys[sysname].add(int(gidx))

    #For this cluster
    cell_ids_arr = np.asarray(cell_ids).astype(np.uint64)
    layer_of_hit = ((cell_ids_arr >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
    locs_arr = np.asarray(locs)
    ws_arr = np.asarray(ws)

    layer_energy = {} #For our cluster getting the energy in each layer
    for sysname, lay, w in zip(locs_arr, layer_of_hit, ws_arr):
        key = (sysname, int(lay))
        layer_energy[key] = layer_energy.get(key, 0.0) + w
    #gets an array for detector location filling by layer the energy for our cluster

    numerator = 0.0
    denominator = 0.0

    #Fill the position for this event
    for (locname, lay), E_s in layer_energy.items(): #going through each layer for our known cluster
        if E_s <= 0:
            continue
        sysname = loc2sys.get(locname)
        if sysname is None or sysname not in real_systems:
            continue
        
        full_cellid = np.asarray(cellid[sysname][event_index]).astype(np.uint64)
        full_layer = ((full_cellid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64)
        #["x"][i][idxs]
        pos_x = np.asarray(pos[sysname]["x"][event_index])
        pos_y = np.asarray(pos[sysname]["y"][event_index])
        pos_z = np.asarray(pos[sysname]["z"][event_index])
        
        layer_mask = full_layer == lay
        layer_global_idx = np.nonzero(layer_mask)[0]
        #use layer_mask, should work on the hits since they're made together

        #exclude our clusters hits

        our_idx = np.array([gi for gi in layer_global_idx if gi in own_idx_by_sys[sysname]], dtype=int)
        candidate_idx = np.array([gi for gi in layer_global_idx if gi not in own_idx_by_sys[sysname]], dtype=int)

        if len(candidate_idx) == 0:
            continue
        
        pos_x_ourclus = pos_x[our_idx]
        pos_y_ourclus = pos_y[our_idx]
        pos_z_ourclus = pos_z[our_idx]

        pos_x_notclus = pos_x[candidate_idx]
        pos_y_notclus = pos_y[candidate_idx]
        pos_z_notclus = pos_z[candidate_idx]

        unclaimed_set = set(unclaimed[sysname]) #just turns into a set
        phi = np.arctan2(pos_y_notclus, pos_x_notclus)
        r_xy = np.sqrt(pos_x_notclus**2 + pos_y_notclus**2)
        theta = np.arctan2(r_xy, pos_z_notclus)
        eta = -np.log(np.tan(theta / 2))

        #Now we have to check to make sure the hits fall into the correct radius

        full_collection = np.zeros(len(pos_x_notclus))
        for index in range(len(pos_x_ourclus)):

            #Will this work without looping (delta x will be an array)

            delta_eta = eta - bary_eta
            dphi = phi - bary_phi
            delta_phi = np.arctan2(np.sin(dphi), np.cos(dphi))
            rs = np.sqrt(delta_eta**2 + delta_phi**2)


            full_collection[rs < some_width] = 1
        
        full_collection = full_collection.astype(bool)
        nearby_idx = candidate_idx[full_collection]
        n_noclus = sum(1 for gi in nearby_idx if int(gi) in unclaimed_set)
        n_neighbor = len(nearby_idx)
        if n_neighbor == 0:
            continue

        numerator += E_s * (n_noclus / n_neighbor)
        denominator += E_s
    
    if numerator is None:
        return None
    if denominator is None:
        return None
    return numerator / denominator if denominator > 0 else None



def cluster_matching(i, bound):
    # Final-state MC particles
    mask = (status_mc[i] == 1)
    mx = np.asarray(mc_x[i])[mask]
    my = np.asarray(mc_y[i])[mask]
    mz = np.asarray(mc_z[i])[mask]
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
    return int(clus_ids[best])

    
    






def plot_isolation_histogram(isolation_dict, bins=50, save_prefix="isolation", particle_name="", color="green"):
    for var_name, values in isolation_dict.items():
        clean_vals = [v for v in values if v is not None and np.isfinite(v)]
        if len(clean_vals) == 0:
            print(f"Skipping {var_name}: no valid data points.")
            continue

        plt.figure(figsize=(7, 5))
        plt.hist(clean_vals, bins=bins, color=color, edgecolor="black", alpha=0.75)
        plt.title(f"Distribution of {var_name} - {particle_name}", fontsize=12)
        plt.xlabel(var_name, fontsize=10)
        plt.ylabel("Counts", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{var_name}_{particle_name}.png", dpi=150)
        plt.close()
        print(f"Saved: {save_prefix}_{var_name}_{particle_name}.png")





# --- 2. Plotting Function ---
def save_histogram(data, label, filename, color):
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=50, color=color, edgecolor='black', alpha=0.7)
    plt.title(f"Calorimeter Entry Point Distribution: {label}")
    plt.xlabel(f"{label} Position [mm]")
    plt.ylabel("Counts")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close() # Free memory
    print(f"Saved: {filename}")


#Now I want one giant plotting function
def plot_combined_histograms(all_particle_data, variable, bins=50, save_prefix="combined", density = False):

    plt.figure(figsize=(7, 5))

    # First collect all valid values from ALL particles
    all_values = []

    for pdg, particle_info in all_particle_data.items():

        values = particle_info["data"][variable]

        clean_vals = [
            v for v in values
            if v is not None and np.isfinite(v)
        ]

        all_values.extend(clean_vals)

    # Make sure we actually have data
    if len(all_values) == 0:
        print(f"Skipping {variable}: no valid data.")
        plt.close()
        return

    # Find the minimum and maximum across ALL particles
    min_value = np.min(all_values)
    max_value = np.max(all_values)

    # Create 50 common bins between the global min and max
    bin_edges = np.linspace(min_value, max_value, bins + 1)

    # Now plot every particle using the SAME bins
    for pdg, particle_info in all_particle_data.items():

        values = particle_info["data"][variable]

        clean_vals = [
            v for v in values
            if v is not None and np.isfinite(v)
        ]

        if len(clean_vals) == 0:
            continue

        plt.hist(
            clean_vals,
            bins=bin_edges,       # <-- same bins for every particle
            color=particle_info["color"],
            alpha=0.5,
            edgecolor="black",
            label=particle_info["name"],
            density = density
        )

    plt.title(f"Distribution of {variable}")
    plt.xlabel(variable)
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(
        f"{save_prefix}_{variable}.png",
        dpi=150
    )

    plt.close()

    print(f"Saved: {save_prefix}_{variable}.png")


def plot_combined_histograms_logbins(all_particle_data, variable, bins=50, save_prefix="combined"):

    plt.figure(figsize=(7, 5))

    # First collect all valid values from ALL particles
    all_values = []

    for pdg, particle_info in all_particle_data.items():

        values = particle_info["data"][variable]

        clean_vals = [
            v for v in values
            if v is not None and np.isfinite(v)
        ]

        all_values.extend(clean_vals)

    # Make sure we actually have data
    if len(all_values) == 0:
        print(f"Skipping {variable} (logscale): no positive values to plot.")
        plt.close()
        return

    # Find the minimum and maximum across ALL particles
    all_values = [v for v in all_values if v > 0]
    if len(all_values) == 0:
        print(f"Skipping {variable} (logscale): no positive values to plot.")
        return
    min_value = np.min(all_values)
    max_value = np.max(all_values)

    # Create 50 common bins between the global min and max
    bin_edges = np.logspace(np.log10(min_value), np.log10(max_value), bins + 1)

    # Now plot every particle using the SAME bins
    for pdg, particle_info in all_particle_data.items():

        values = particle_info["data"][variable]

        clean_vals = [
            v for v in values
            if v is not None and np.isfinite(v) and v > 0
        ]

        if len(clean_vals) == 0:
            continue

        plt.hist(
            clean_vals,
            bins=bin_edges,       # <-- same bins for every particle
            color=particle_info["color"],
            alpha=0.5,
            edgecolor="black",
            label=particle_info["name"]
        )

    plt.title(f"Distribution of {variable}")
    plt.xlabel(variable)
    plt.xscale("log")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(
        f"{save_prefix}_{variable}_logscale.png",
        dpi=150
    )

    plt.close()

    print(f"Saved: {save_prefix}_{variable}_logscale.png")

def plot_signal_core(signal, hcalfrac, particle_name, title, name):
    plt.figure(figsize=(8, 6))

    plt.hist2d(
        hcalfrac,
        signal,
        bins=(40, 40),
        cmap="viridis"
    )

    plt.colorbar(label="Number of events")

    plt.xlabel("HCAL Layer Fraction")
    plt.ylabel(f"{title}")
    plt.title(f"{title} vs HCAL Fraction for {particle_name}")

    plt.tight_layout()
    plt.savefig(f"core_signal_fraction_{name}_{particle_name}.png")
    plt.close()

def plot_2d_hist(item1, item2, particle_name, title, name, item1name, item2name):
    item1 = np.asarray(item1)
    item2 = np.asarray(item2)

    plt.figure(figsize=(8, 6))

    xbins = np.arange(item1.min() - 0.5, item1.max() + 1.5, 1)  # one bin per cluster count
    ybins = 50                                                   # isolation is continuous

    plt.hist2d(
        item1, item2,                 # x = item1, y = item2 (matches the labels)
        bins=(xbins, ybins),
        cmap="viridis",
        norm=LogNorm(),
        cmin=1,
    )
    plt.colorbar(label="Number of events (log scale)")

    plt.xlabel(item1name)
    plt.ylabel(item2name)
    plt.title(f"{title} for {particle_name}")
    plt.tight_layout()
    plt.savefig(f"{item1name}_{item2name}_{particle_name}.png", dpi=150)
    plt.close()

# Labels for cluster moments
labels_moments = [
    r"$\eta_{\rm bary}$",
    "Radial first",
    "Longitudinal calorimeter first",
    r"$\theta_{\rm RMS}$",
    "Radial RMS",
    "Longitudinal RMS",
    "Radial Skewness",
    "Longitudinal Skewness",
    "Radial Kurtosis",
    "Longitudinal Kurtosis"
]


# Labels for time moments
labels_time = [
    "Longitudinal calorimeter first",
    "Radial first",
    r"$\theta_{\rm RMS}$",
    "Radial RMS",
    "Longitudinal RMS",
    "Time energy-weighted center",
    "Time variance",
    "Time asymmetry",
    "Time tails"
]


# Labels for angularity
labels_angularity = [
    r"$z_1,\ \alpha=0.5$",
    r"$z_2,\ \alpha=1$",
    r"$z_2,\ \alpha=2$",
    r"$z_3,\ \alpha=1$",
    r"$z_3,\ \alpha=2$",
    "Longitudinal calorimeter first",
    "Radial first",
    r"$\theta_{\rm RMS}$"
]


# Labels for signal composition
labels_signalcomp = [
    "EM fraction",
    "HAD fraction",
    "Most energetic hit",
    "Most energetic HAD hit",
    "Core signal fraction"
]


# Labels for isolation
labels_isolation = [
    "Mean isolation",
    "Least isolation",
    "Most isolation",
    "Mean hit isolation",
    "Least hit isolation",
    "Most hit isolation",
    "Mean angular isolation",
    "Least angular isolation",
    "Most angular isolation"
]

def corner_and_corr(pdf, data, labels, title, color, pname, method="spearman"):
    data = np.asarray(data, dtype=float)
    n_before = len(data)
    data = data[np.all(np.isfinite(data), axis=1)]
    print(f"[{pname}] {title}: {n_before} rows -> {len(data)} after dropping NaN/inf")

    # drop constant columns (corner crashes on them)
    keep = [k for k in range(data.shape[1]) if np.ptp(data[:, k]) > 0]
    if len(keep) < 2 or len(data) < 10:
        print(f"[{pname}] {title}: skipped, not enough data")
        return
    data = data[:, keep]
    labels = [labels[k] for k in keep]

    # clip axes to 0.5-99.5 percentiles so outliers don't squash everything
    rng = [(np.percentile(c, 0.5), np.percentile(c, 99.5)) for c in data.T]
    rng = [(lo, hi) if hi > lo else (lo - 0.5, hi + 0.5) for lo, hi in rng]

    fig = corner.corner(
        data, labels=labels, color=color, range=rng,
        bins=25, smooth=1.0,
        plot_density=False, plot_datapoints=True, fill_contours=True,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)),
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt=".2g",
        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 8},
    )
    fig.suptitle(f"{pname} - {title}  (N = {len(data)})", y=1.02)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # correlation matrix for the same group
    corr = pd.DataFrame(data, columns=labels).corr(method=method)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * n), max(5, 0.7 * n)))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, ax=ax)
    ax.set_title(f"{pname} - {title} correlation ({method})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

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

    calo_faces_x = []
    calo_faces_y = []
    calo_faces_z = []
    for index in range(events.num_entries):
        for pfo in range(len(pfo_energy[index])):
            cluster_begin = pandora_clusters_begin[index][pfo]
            cluster_end = pandora_clusters_end[index][pfo]
            tracks_begin = pandora_tracks_begin[index][pfo]
            tracks_end = pandora_tracks_end[index][pfo]
            if (tracks_begin != tracks_end):
                for track in range (tracks_begin, tracks_end): #this is most nearly definitely only 1 track
                    track_index = pandora_tracks_index[index][track]
                    x_entry, y_entry, z_entry = get_entry_point_from_track(index, track_index)
                    if x_entry is not None:
                        calo_faces_x.append(x_entry)
                        calo_faces_y.append(y_entry)
                        calo_faces_z.append(z_entry)

    #Now I want to plot each 3 on the graph to check the distribution
    calo_faces_x = np.array(calo_faces_x)
    calo_faces_y = np.array(calo_faces_y)
    calo_faces_z = np.array(calo_faces_z)

    # --- 3. Save the 3 PNG Plots ---
    save_histogram(calo_faces_x, "X", "calo_face_x.png", "royalblue")
    save_histogram(calo_faces_y, "Y", "calo_face_y.png", "forestgreen")
    save_histogram(calo_faces_z, "Z", "calo_face_z.png", "crimson")


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
    }

    cluster_data_t = {
        "barycenters_eta_time": [],
        "barycenters_phi_time": [],
        "cog_x_time": [],
        "cog_y_time": [],
        "cog_z_time": [],
        "long_first_time": [],
        "radial_first_time": [],
        "longcalo_first_time": [],
        "semi_length_L2_time": [],
        "semi_length_T2_time": [],
        "width_measure_time": [],
        "length_measure_time": [],
        "effective_eta_phi_time": [],
        "zeroth_moment_time": [],
        "aspect_ratios_time": [],
        "theta_rms_all_time": [],
        "radial_rms_all_time": [],
        "longitudinal_rms_all_time": [],
        "r_thirdmoment_time": [],
        "r_fourthmoment_time": [],
        "r_skewness_time": [],
        "r_kurtosis_time": [],
        "l_thirdmoment_time": [],
        "l_fourthmoment_time": [],
        "l_skewness_time": [],
        "l_kurtosis_time": [],
    }

    cluster_data_time = {
        "time_energyweighted_center": [],
        "time_center": [],
        "time_moment2_variance": [],
        "time_moment2_variance_sr": [],
        "time_moment3_asymm": [],
        "time_moment4_tails": [],

    }
    cluster_data_signalcomp = {
        "EM_fraction": [],
        "HAD_fraction": [],
        "most_energetic_hit_all": [],
        "most_energetic_hit_em": [],
        "most_energetic_hit_had": [],
        "core_signal_frac_all": [],
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
        "z_3_alpha_2": []
    }
    cluster_data_extra = {
        "core_signal": [],
        "hcal_ratio": [],
        "hcal_over_all": [],
        "ecal_over_all": [],
        "hcal_over_hcal": [],
        "ecal_over_ecal": []
    }
    isolation_values = {
        "mean_iso": [],
        "least_iso": [],
        "most_iso": [],
        "mean_iso_hits": [],
        "least_iso_hits": [],
        "most_iso_hits": [],
        "mean_angular": [],
        "least_angular": [],
        "most_angular": [],
        "mean_angular_hits": [],
        "most_angular_hits": [],
        "least_angular_hits": []
    }
    isolation_matching = {
        "isolation_match": [],
        "isolation_angular": [],
        "isolation_hits": [],
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


    def stats(lst):
        return (np.mean(lst), min(lst), max(lst)) if lst else (np.nan, np.nan, np.nan)

    angular_est = 0.15
    '''
    for event_index in range(events.num_entries):
        n_clusters = len(cluster_energy[event_index])
        if n_clusters == 1:
            continue
        unclaimed = unclaimed_indexes(event_index, n_clusters)
        all_list, all_list_hits, all_list_ang, all_list_anghit = [], [], [], []

        for cluster_index in range(n_clusters):
            result = return_barycenter_centre_of_gravity(event_index, cluster_index)
            if result is None:
                continue
            eta_bary, phi_bary, _, _, _, _, ws, id_cell, locs = result

            iso   = isolation(unclaimed, cluster_index, event_index, ws, id_cell, locs, some_width)
            iso_h = isolation_allhits(cluster_index, event_index, ws, id_cell, locs, some_width)
            iso_h_etaphi = isolation_allhits_etaphi(cluster_index, event_index, ws, id_cell, locs, angular_est, eta_bary, phi_bary)
            iso_a = isolation_eta_phi(unclaimed, cluster_index, event_index, ws, id_cell, locs,
                                      angular_est, eta_bary, phi_bary)
            if iso is not None:
                all_list.append(iso)
                isolation_alone["isolation_all"].append(iso)
            if iso_h is not None:
                all_list_hits.append(iso_h)
                isolation_alone["isolation_hits"].append(iso_h)
            if iso_a is not None:
                all_list_ang.append(iso_a)
                isolation_alone["isolation_angular"].append(iso_a)
            if iso_h_etaphi is not None:
                all_list_anghit.append(iso_h_etaphi)
                isolation_alone["isolation_hits_angular"].append(iso_h_etaphi)

        m, l, M    = stats(all_list)
        mh, lh, Mh = stats(all_list_hits)
        ma, la, Ma = stats(all_list_ang)
        mha, lha, Mha = stats(all_list_anghit)

        isolation_values["mean_iso"].append(m)
        isolation_values["least_iso"].append(l)
        isolation_values["most_iso"].append(M)
        isolation_values["mean_iso_hits"].append(mh)
        isolation_values["least_iso_hits"].append(lh)
        isolation_values["most_iso_hits"].append(Mh)
        isolation_values["mean_angular"].append(ma)
        isolation_values["least_angular"].append(la)
        isolation_values["most_angular"].append(Ma)
        isolation_values["least_angular_hits"].append(lha)
        isolation_values["most_angular_hits"].append(Mha)
        isolation_values["mean_angular_hits"].append(mha)

        cluster_list.append(n_clusters)
    def plot_2d_hist_clean(x, y, particle_name, title, name, xname, yname):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if not good.any():
            print(f"Skipping {yname}: no valid points.")
            return
        plot_2d_hist(x[good], y[good], particle_name, title, name, xname, yname)

    pname = particle_info["name"]

    plot_2d_hist_clean(cluster_list, isolation_values["least_iso"], pname, "title", "Number of Clusters vs Least Isolated Isolation", "Num_clusters", "Least_Isolation")
    plot_2d_hist_clean(cluster_list, isolation_values["most_iso"], pname, "title", "Number of Clusters vs Most Isolated Isolation", "Num_clusters", "Most_Isolation")
    plot_2d_hist_clean(cluster_list, isolation_values["mean_iso"], pname, "title", "Number of Clusters vs Mean Isolation", "Num_clusters", "Mean_Isolation")

    plot_2d_hist_clean(cluster_list, isolation_values["least_iso_hits"], pname, "title", "Number of Clusters vs Least Isolated Isolation Hits", "Num_clusters", "Least_Isolation_Hits")
    plot_2d_hist_clean(cluster_list, isolation_values["most_iso_hits"], pname, "title", "Number of Clusters vs Most Isolated Isolation Hits", "Num_clusters", "Most_Isolation_Hits")
    plot_2d_hist_clean(cluster_list, isolation_values["mean_iso_hits"], pname, "title", "Number of Clusters vs Mean Isolation Hits", "Num_clusters", "Mean_Isolation_Hits")

    plot_2d_hist_clean(cluster_list, isolation_values["least_angular"], pname, "Angular = 0.15", "Number of Clusters vs Least Isolated Isolation Angular", "Num_clusters", "Least_Isolation_Angular")
    plot_2d_hist_clean(cluster_list, isolation_values["most_angular"], pname, "Angular = 0.15", "Number of Clusters vs Most Isolated Isolation Angular", "Num_clusters", "Most_Isolation_Angular")
    plot_2d_hist_clean(cluster_list, isolation_values["mean_angular"], pname, "Angular = 0.15", "Number of Clusters vs Mean Isolation Angular", "Num_clusters", "Mean_Isolation_Angular")

    plot_2d_hist_clean(cluster_list, isolation_values["least_angular_hits"], pname, "Angular = 0.15", "Number of Clusters vs Least Isolated Isolation Angular Hits", "Num_clusters", "Least_Isolation_Angular_Hits")
    plot_2d_hist_clean(cluster_list, isolation_values["most_angular_hits"], pname, "Angular = 0.15", "Number of Clusters vs Most Isolated Isolation Angular Hits", "Num_clusters", "Most_Isolation_Angular_Hits")
    plot_2d_hist_clean(cluster_list, isolation_values["mean_angular_hits"], pname, "Angular = 0.15", "Number of Clusters vs Mean Isolation Angular Hits", "Num_clusters", "Mean_Isolation_Angular_Hits")

    plot_isolation_histogram(
        {"isolation_all": isolation_alone["isolation_all"]},
        bins=50,
        save_prefix="isolation",
        particle_name=pname,
        color=particle_info["color"]
    )

    all_isolation_data[pdg] = {
        "name": particle_info["name"],
        "color": particle_info["color"],
        "data": {
            "mean_iso": isolation_values["mean_iso"],
            "least_iso": isolation_values["least_iso"],
            "most_iso": isolation_values["most_iso"],
            "isolation_all": isolation_alone["isolation_all"],
            "mean_iso_hits": isolation_values["mean_iso_hits"],
            "least_iso_hits": isolation_values["least_iso_hits"],
            "most_iso_hits": isolation_values["most_iso_hits"],
            "isolation_hits_all": isolation_alone["isolation_hits"],
            "mean_iso_angular": isolation_values["mean_angular"],
            "least_iso_angular": isolation_values["least_angular"],
            "most_iso_angular": isolation_values["most_angular"],
            "most_iso_angularhits": isolation_values["mean_angular_hits"],
            "least_iso_angularhits": isolation_values["least_angular_hits"],
            "mean_iso_angularhits": isolation_values["most_angular_hits"]
        },
    }
    plt.figure(figsize=(8, 6))
    '''

    #Meow, meow  
    #Now I want to do it with just the "matched clusters"
    cluster_axis_length = []
    cluster_clus_iso = []
    cluster_all_iso = []
    cluster_energy_list = []

    for event_index in range(events.num_entries):
        n_clusters = len(cluster_energy[event_index])
        if n_clusters == 1:
            continue
        #clusters_begin = pandora_clusters_begin[event_index]
        #clusters_end = pandora_clusters_end[event_index]
        #length_clusters = len(clusters_begin)
        unclaimed = unclaimed_indexes(event_index, n_clusters)
        all_list = []
        least = 0
        most = 0
        isolation_check = False
        matched_idx = cluster_matching(event_index, particle_info['matching_bound'])
        if (matched_idx is not None):
            result = return_barycenter_centre_of_gravity(event_index, matched_idx)
            if result is None:
                continue
            eta_bary, phi_bary, [cogx, cogy, cogz], xs, ys, zs, ws, id_cell, locs = result
            x, xx, xx, xxx, sigma_Tsq, xxxx, xxxxx, xxxxxx = get_shower_axis(xs, ys, zs, ws, [cogx, cogy, cogz])
            isolation_num = isolation(unclaimed, matched_idx, event_index, ws, id_cell, locs, some_width)
            sum_energy = np.sum(ws)
            isolation_hits = isolation_allhits(matched_idx, event_index, ws, id_cell, locs, some_width) 
            isolation_angular = isolation_eta_phi(unclaimed, matched_idx, event_index, ws, id_cell, locs, angular_est, eta_bary, phi_bary)
            isolation_but_all, isolation_but_clus = isolation_allhits_but(unclaimed, matched_idx, event_index, ws, id_cell, locs, sigma_Tsq / 2)

            if isolation_num is None:
                continue
            isolation_matching["isolation_match"].append(isolation_num)
            isolation_matching["isolation_hits"].append(isolation_hits)
            isolation_matching["isolation_angular"].append(isolation_angular)
            isolation_matching["isolation_hits_but_all"].append(isolation_but_all)
            isolation_matching["isolation_hits_but_clus"].append(isolation_but_clus)
            cluster_axis_length.append(sigma_Tsq)
            cluster_clus_iso.append(isolation_but_clus)
            cluster_all_iso.append(isolation_but_all)
            cluster_energy_list.append(sum_energy)
    
    #Now let us make a heat graph together
    axis = np.array([a if a is not None else np.nan for a in cluster_axis_length], dtype=float)
    clus = np.array([v if v is not None else np.nan for v in cluster_clus_iso], dtype=float)
    allv = np.array([v if v is not None else np.nan for v in cluster_all_iso], dtype=float)
    enerv = np.array([v if v is not None else np.nan for v in cluster_energy_list], dtype=float)

    pname = particle_info["name"]

    '''
    plots = [
        (clus, "Isolation (clustered neighbors only)", "clus"),
        (allv, "Isolation (all nearby hits)",          "all"),
    ]

    for vals, label, tag in plots:
        ok = np.isfinite(axis) & np.isfinite(vals)

        fig, ax = plt.subplots(figsize=(6, 5))
        h = ax.hist2d(axis[ok], vals[ok], bins=40, cmap="viridis", cmin=1)
        fig.colorbar(h[3], ax=ax, label="clusters")
        ax.set_xlabel(r"Cluster axis length $\sigma_T^2$")
        ax.set_ylabel(label)
        ax.set_title(f"{pname}: {label}\n(N = {ok.sum()})")

        fig.tight_layout()
        fig.savefig(f"{pname}_{tag}_iso_vs_axis.png", dpi=200)
        plt.show()
    '''
    plots = [
        (clus, "Isolation (clustered neighbors only)", "clus"),
        (allv, "Isolation (all nearby hits)",          "all"),
    ]

    for vals, label, tag in plots:
        ok = np.isfinite(enerv) & np.isfinite(vals)

        fig, ax = plt.subplots(figsize=(6, 5))
        h = ax.hist2d(enerv[ok], vals[ok], bins=40, cmap="viridis", cmin=1)
        fig.colorbar(h[3], ax=ax, label="clusters")
        ax.set_xlabel(r"Cluster Energy")
        ax.set_ylabel(label)
        ax.set_title(f"{pname}: {label}\n(N = {ok.sum()})")

        fig.tight_layout()
        fig.savefig(f"{pname}_{tag}_iso_vs_energy.png", dpi=200)
        plt.show()

    

    all_matched_iso[pdg] = {
        "name": particle_info["name"],
        "color": particle_info["color"],
        "data": {
            "matched_isolation": isolation_matching["isolation_match"],
            "matched_isolation_hits": isolation_matching["isolation_hits"],
            "matched_isolation_angular": isolation_matching["isolation_angular"],
            "isolation_hits_but_all": isolation_matching["isolation_hits_but_all"],
            "isolation_hits_but_clus": isolation_matching["isolation_hits_but_clus"]
        },
    }
    
    
    
            
    for index in range(events.num_entries):
        unclaimed = unclaimed_indexes(index, len(cluster_energy[index]))
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
                    total_energy = zeroth_moment(index, cluster_index)
                    eta_bary, phi_bary, [cog_x, cog_y, cog_z], xs, ys, zs, ws, id_cell, locs = return_barycenter_centre_of_gravity(index, cluster_index)
                    core_signal_frac = core_signal_fraction(ws, id_cell, locs)
                    core_signal_frac2, hcalratio, hcal_over_all, ecal_over_all, hcal_over_hcal, ecal_over_ecal = core_signal_fraction_detector(ws, id_cell, locs)
                    cluster_data_extra["core_signal"].append(core_signal_frac2)
                    cluster_data_extra["hcal_ratio"].append(hcalratio)
                    cluster_data_extra["hcal_over_all"].append(hcal_over_all)
                    cluster_data_extra["ecal_over_all"].append(ecal_over_all)
                    cluster_data_extra["ecal_over_ecal"].append(ecal_over_ecal)
                    cluster_data_extra["hcal_over_hcal"].append(hcal_over_hcal)
                    axis, sigma_L2, sigma_T2, sigma_L2sq, sigma_T2sq, theta_rms, _moment_ratio, aspect_ratio= get_shower_axis(xs, ys, zs, ws, [cog_x, cog_y, cog_z])
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
                    
                    cluster_data["zeroth_moment"].append(total_energy)
                    cluster_data["barycenters_eta"].append(eta_bary)
                    cluster_data["barycenters_phi"].append(phi_bary)
                    cluster_data["cog_x"].append(cog_x)
                    cluster_data["cog_y"].append(cog_y)
                    cluster_data["cog_z"].append(cog_z)
                    cluster_data["radial_first"].append(r_firstmoment)
                    cluster_data["long_first"].append(l_firstmoment)
                    cluster_data["longcalo_first"].append(l_firstmoment_entry)
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

                    cluster_data_signalcomp["core_signal_frac_all"].append(core_signal_frac)
                    #cluster_data_signalcomp["isolation_all"].append(isolation_num)
    #plot_signal_core(cluster_data_extra["ecal_over_ecal"], cluster_data_extra["hcal_ratio"], particle_info["name"], "ECAL / ECAL Fraction", "ECAL")
    #plot_signal_core(cluster_data_extra["hcal_over_hcal"], cluster_data_extra["hcal_ratio"], particle_info["name"], "HCAL / HCAL Fraction", "HCAL")



    #Let's have a plotting function
    #Let's just make a basic rudimentary plot for all of these, bar plot counts vs whatever variable this is
    #plot_cluster_variable(cluster_data, bins=50, save_prefix="moment_spatial", particle_name=particle_info["name"], color=particle_info["color"])

    #Now I want to do them with time!:
    all_hit_times = []
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
                    total_time = zeroth_moment_time(index, cluster_index)
                    eta_bary, phi_bary, [cog_x, cog_y, cog_z], xs, ys, zs, ws = return_barycenter_centre_of_time(index, cluster_index)
                    dontcare1, dontcare2, [dontcare3, dontcare4, dontcare5], dontcare6, dontcare7, dontcare8, energy, dontcare9, dontcare10 = return_barycenter_centre_of_gravity(index, cluster_index)
                    #Now let's get a histogram for ws where ws is ts, then also energy weighted for time
                    #ws here is time
                    all_hit_times.extend(ws)


                    axis, sigma_L2, sigma_T2, sigma_L2sq, sigma_T2sq, theta_rms, _moment_ratio, aspect_ratio= get_shower_axis(xs, ys, zs, ws, [cog_x, cog_y, cog_z])
                    rms_radial, rms_longitudinal, r_firstmoment, l_firstmoment, l_firstmoment_entry = get_rms_cluster_dimensions(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis, entry_point)
                    m2_lat, m2_long = get_normalized_energy_disp(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis)
                    sigma_eta_phi = get_effective_cluster_size(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis, eta_bary, phi_bary)
                    r_thirdmoment, l_thirdmoment, r_fourthmoment, l_fourthmoment, r_skewness, l_skewness, r_kurtosis, l_kurtosis = get_third_fourth_moments(xs, ys, zs, ws, [cog_x, cog_y, cog_z], axis)
                    cluster_data_t["zeroth_moment_time"].append(total_time)
                    cluster_data_t["barycenters_eta_time"].append(eta_bary)
                    cluster_data_t["barycenters_phi_time"].append(phi_bary)
                    cluster_data_t["cog_x_time"].append(cog_x)
                    cluster_data_t["cog_y_time"].append(cog_y)
                    cluster_data_t["cog_z_time"].append(cog_z)
                    cluster_data_t["radial_first_time"].append(r_firstmoment)
                    cluster_data_t["long_first_time"].append(l_firstmoment)
                    cluster_data_t["longcalo_first_time"].append(l_firstmoment_entry)
                    cluster_data_t["semi_length_L2_time"].append(sigma_L2)
                    cluster_data_t["semi_length_T2_time"].append(sigma_T2)
                    cluster_data_t["radial_rms_all_time"].append(rms_radial)
                    cluster_data_t["longitudinal_rms_all_time"].append(rms_longitudinal)
                    cluster_data_t["theta_rms_all_time"].append(theta_rms)
                    cluster_data_t["aspect_ratios_time"].append(aspect_ratio)
                    cluster_data_t["effective_eta_phi_time"].append(sigma_eta_phi)
                    cluster_data_t["width_measure_time"].append(m2_lat)
                    cluster_data_t["length_measure_time"].append(m2_long)
                    cluster_data_t["r_thirdmoment_time"].append(r_thirdmoment)
                    cluster_data_t["r_fourthmoment_time"].append(r_fourthmoment)
                    cluster_data_t["r_skewness_time"].append(r_skewness)
                    cluster_data_t["r_kurtosis_time"].append(r_kurtosis)
                    cluster_data_t["l_thirdmoment_time"].append(l_thirdmoment)
                    cluster_data_t["l_fourthmoment_time"].append(l_fourthmoment)
                    cluster_data_t["l_skewness_time"].append(l_skewness)
                    cluster_data_t["l_kurtosis_time"].append(l_kurtosis)
    
    plt.figure(figsize=(7, 5))

    bin_edges = np.linspace(-0.1, 0.5, 51)
    plt.hist(
        all_hit_times,
        bins=bin_edges,
        edgecolor="black"
    )

    plt.yscale("log")
    plt.xlabel("Hit time")
    plt.ylabel("Counts")
    plt.title(f"Hit Time Distribution - {particle_info['name']}")

    plt.tight_layout()

    plt.savefig(
        f"hit_time_distribution_{particle_info['name']}.png",
        dpi=150
    )

    plt.close()


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

    #plot_cluster_variable(cluster_data_signalcomp, bins=50, save_prefix="moment_signal", particle_name=particle_info["name"], color=particle_info["color"])
    '''
    all_data = {}
    all_data.update(cluster_data)
    all_data.update(cluster_data_t)
    all_data.update(cluster_data_signalcomp)
    all_data.update(cluster_data_angularity)

    df = pd.DataFrame(all_data)

    variables_for_corr = [
        "radial_first",
        "longcalo_first",
        "semi_length_L2",
        "semi_length_T2",
        "width_measure",
        "length_measure",
        "effective_eta_phi",
        "radial_rms_all",
        "longitudinal_rms_all",
        "l_thirdmoment",
        "r_thirdmoment",
        "l_fourthmoment",
        "r_fourthmoment",

        "radial_first_time",
        "longcalo_first_time",
        "semi_length_L2_time",
        "semi_length_T2_time",
        "width_measure_time",
        "length_measure_time",
        "effective_eta_phi_time",
        "radial_rms_all_time",
        "longitudinal_rms_all_time",
        "r_skewness_time",
        "l_skewness_time",

        "EM_fraction",
        "HAD_fraction",
        "most_energetic_hit_all",
        "most_energetic_hit_em",
        "most_energetic_hit_had",
    ]
    
    corr = df[variables_for_corr].corr()
    corr = df[variables_for_corr].corr()

    plt.figure(figsize=(20, 18))

    print("cluster_data:", {key: len(value) for key, value in cluster_data.items()})
    print("cluster_data_t:", {key: len(value) for key, value in cluster_data_t.items()})
    print("cluster_data_signalcomp:", {key: len(value) for key, value in cluster_data_signalcomp.items()})
    print("cluster_data_time:", {key: len(value) for key, value in cluster_data_time.items()})
    print("cluster_data_angularity:", {key: len(value) for key, value in cluster_data_angularity.items()})
    df_all = pd.DataFrame({ #This si how we include all of them at once
        **cluster_data,
        **cluster_data_t,
        **cluster_data_signalcomp,
        **cluster_data_time,
        **cluster_data_angularity
    })


    all_variables = df_all.columns.tolist()

    df_time = pd.DataFrame({
        **cluster_data_time
    })

    all_time = df_time.columns.tolist()

    all_time_data[pdg] = {
        "name": particle_info["name"],
        "color": particle_info["color"],
        "data": df_time
    }

    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0
    )

    plt.title(f"Correlation Matrix - {particle_info['name']}")
    plt.tight_layout()

    plt.savefig(
        f"correlation_matrix_{particle_info['name']}.png",
        dpi=150
    )

    plt.close()
    all_particle_data[pdg] = {
        "name": particle_info["name"],
        "color": particle_info["color"],
        "data": df_all
    }



    #Now let's build the corner plots:
        #Should make the corner plots now
    #Cluster data time
    #Cluster data angularity
    # Cluster data signalcomp
    # Cluster data
    cluster_moments_clean = np.column_stack([cluster_data["barycenters_eta"], cluster_data["radial_first"], cluster_data["longcalo_first"], cluster_data["theta_rms_all"], cluster_data["radial_rms_all"], cluster_data["longitudinal_rms_all"], cluster_data["r_skewness"], cluster_data["l_skewness"], cluster_data["r_kurtosis"], cluster_data["l_kurtosis"]])
    cluster_moments_clean = np.asarray(cluster_moments_clean, dtype=float)
    cluster_moments_clean = cluster_moments_clean[
        np.all(np.isfinite(cluster_moments_clean), axis=1)
    ]
    #cluster_moments_clean = cluster_moments[np.all(np.isfinite(cluster_moments), axis = 1)]
    cluster_time_moments_clean = np.column_stack([cluster_data["longcalo_first"], cluster_data["radial_first"], cluster_data["theta_rms_all"], cluster_data["radial_rms_all"], cluster_data["longitudinal_rms_all"], cluster_data_time["time_energyweighted_center"], cluster_data_time["time_moment2_variance"], cluster_data_time["time_moment3_asymm"], cluster_data_time["time_moment4_tails"]])
    cluster_time_moments_clean = np.asarray(cluster_time_moments_clean, dtype=float)
    cluster_time_moments_clean = cluster_time_moments_clean[
        np.all(np.isfinite(cluster_time_moments_clean), axis = 1)
    ]
    #cluster_time_moments_clean = cluster_time_moments[np.all(np.isfinite(cluster_time_moments), axis = 1)]
    cluster_angularity_clean = np.column_stack([cluster_data_angularity["z_1_alpha_0_5"], cluster_data_angularity["z_2_alpha_1"], cluster_data_angularity["z_2_alpha_2"], cluster_data_angularity["z_3_alpha_1"], cluster_data_angularity["z_3_alpha_2"], cluster_data["longcalo_first"], cluster_data["radial_first"], cluster_data["theta_rms_all"]])
    cluster_angularity_clean = np.asarray(cluster_angularity_clean, dtype=float)
    cluster_angularity_clean = cluster_angularity_clean[
        np.all(np.isfinite(cluster_angularity_clean), axis = 1)
    ]
    #cluster_angularity_clean = cluster_angularity[np.all(np.isfinite(cluster_angularity), axis = 1)]
    cluster_signalcomp_clean = np.column_stack([cluster_data_signalcomp["EM_fraction"], cluster_data_signalcomp["HAD_fraction"], cluster_data_signalcomp["most_energetic_hit_all"], cluster_data_signalcomp["most_energetic_hit_had"], cluster_data_signalcomp["core_signal_frac_all"]])
    cluster_signalcomp_clean = np.asarray(cluster_signalcomp_clean, dtype=float)
    cluster_signalcomp_clean = cluster_signalcomp_clean[
        np.all(np.isfinite(cluster_signalcomp_clean), axis = 1)
    ]
    #cluster_signalcomp_clean = cluster_signalcomp[np.all(np.isfinite(cluster_signalcomp), axis = 1)]
    cluster_isolation_clean = np.column_stack([isolation_values["mean_iso"], isolation_values["least_iso"], isolation_values["most_iso"], isolation_values["mean_iso_hits"], isolation_values["least_iso_hits"], isolation_values["most_iso_hits"], isolation_values["mean_angular"], isolation_values["least_angular"], isolation_values["most_angular"]])
    cluster_isolation_clean = np.asarray(cluster_isolation_clean, dtype=float)
    cluster_isolation_clean = cluster_isolation_clean[
        np.all(np.isfinite(cluster_isolation_clean), axis = 1)
    ]
    #cluster_isolation_clean = cluster_isolation[np.all(np.isfinite(cluster_isolation), axis = 1)]

    #Let's put the corner plots together

    pname = particle_info["name"]
    color = particle_info["color"]

    with PdfPages(f"corner_plots_{pname}.pdf") as pdf:   # one file per particle
        corner_and_corr(pdf, cluster_moments_clean,      labels_moments,    "Cluster Moments",    color, pname)
        corner_and_corr(pdf, cluster_time_moments_clean, labels_time,       "Time Moments",       color, pname)
        corner_and_corr(pdf, cluster_angularity_clean,   labels_angularity, "Angularity",         color, pname)
        corner_and_corr(pdf, cluster_signalcomp_clean,   labels_signalcomp, "Signal Composition", color, pname)
        corner_and_corr(pdf, cluster_isolation_clean,    labels_isolation,  "Isolation",          color, pname)
    
    
for variable in ["mean_iso", "least_iso", "most_iso", "isolation_all", "mean_iso_hits", "least_iso_hits", "most_iso_hits", "isolation_hits_all", "mean_iso_angular", "least_iso_angular", "most_iso_angular", "mean_iso_angularhits", "least_iso_angularhits", "most_iso_angularhits"]:
    plot_combined_histograms(
        all_isolation_data,
        variable,
        bins=50,
        save_prefix="combined_isolation",
        density = True
    )
'''
for variable in ["matched_isolation", "matched_isolation_hits", "matched_isolation_angular", "isolation_hits_but_all", "isolation_hits_but_clus"]:
    plot_combined_histograms(
        all_matched_iso,
        variable,
        bins=50,
        save_prefix="combined_isolation_matched",
        density = True
    ) 
#all_matched_iso
'''
for variable in all_variables:
    plot_combined_histograms(
        all_particle_data,
        variable,
        bins=50,
        save_prefix="combined"
    )
'''
'''
for variable in all_time:
    print("Variable:", variable)
    plot_combined_histograms_logbins(
        all_time_data,
        variable,
        bins=50,
        save_prefix="combined"
    )
'''






#Note long second and radial second, is the normalized m2_lat, m2_long if any of these graphs appear
#It is a pure duplicate

#How would I make a loop that would run through 11, 211, 2112 replacing that in events
#I want 11 = green, 211 = royalblue, 2112 = red replacting that in the colors 
#Also it to say electron, pion, neutron = replacing that in the file png name