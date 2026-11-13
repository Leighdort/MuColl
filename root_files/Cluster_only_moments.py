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
import textwrap
import numpy as np
import pandas as pd
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

    def _fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4g}"
        return str(v)
 
 
    def build_entry_point_map(i):
        """cluster_index -> [x, y, z] at the calorimeter face (only for clusters whose PFO has a track)."""
        out = {}
        for pfo in range(len(pfo_energy[i])):
            t0, t1 = pandora_tracks_begin[i][pfo], pandora_tracks_end[i][pfo]
            if t0 == t1:
                continue
            ep = None
            for t in range(t0, t1):
                track_index = pandora_tracks_index[i][t]
                res = get_entry_point_from_track(i, track_index)
                if res is not None:
                    ep = list(res)
            if ep is None:
                continue
            for c in range(pandora_clusters_begin[i][pfo], pandora_clusters_end[i][pfo]):
                out[int(pandora_clusters_index[i][c])] = ep
        return out
    
    
    def compute_cluster_variables(i, j, unclaimed, entry_point, some_width, angular_est):
        d = {}
        errors = []
    
        def run(name, fn):
            try:
                with np.errstate(all="ignore"):
                    fn()
            except Exception as e:
                errors.append(f"{name}: {e!r}")
    
        # --- energy-weighted barycentre (needed by almost everything else) ---
        res = return_barycenter_centre_of_gravity(i, j)
        if res is None:
            return None
        eta_b, phi_b, cog, xs, ys, zs, ws, id_cell, locs = res
        d["bary_eta"], d["bary_phi"] = eta_b, phi_b
        d["cog_x"], d["cog_y"], d["cog_z"] = cog
        d["n_hits"] = len(ws)
        d["energy_total"] = float(np.sum(ws))
        d["has_track_entry"] = entry_point is not None
    
        def shape():
            axis, sL2, sT2, sLsq, sTsq, th_rms, mom_ratio, aspect = get_shower_axis(xs, ys, zs, ws, cog)
            d.update(semi_length_L2=sL2, semi_length_T2=sT2, width_measure=sTsq,
                    length_measure=sLsq, theta_rms=th_rms, moment_ratio=mom_ratio, aspect_ratio=aspect)
            rr, rl, r1, l1, l1e = get_rms_cluster_dimensions(xs, ys, zs, ws, cog, axis, entry_point)
            d.update(radial_rms=rr, longitudinal_rms=rl, radial_first=r1,
                    long_first=l1, longcalo_first=l1e)
            d["m2_lat"], d["m2_long"] = get_normalized_energy_disp(xs, ys, zs, ws, cog, axis)
            d["effective_eta_phi"] = get_effective_cluster_size(xs, ys, zs, ws, cog, axis, eta_b, phi_b)
            (r3, l3, r4, l4, rsk, lsk, rku, lku) = get_third_fourth_moments(xs, ys, zs, ws, cog, axis)
            d.update(r_thirdmoment=r3, l_thirdmoment=l3, r_fourthmoment=r4, l_fourthmoment=l4,
                    r_skewness=rsk, l_skewness=lsk, r_kurtosis=rku, l_kurtosis=lku)
            for key, kap, alp in [("z_1_alpha_0_5", 1.0, 0.5), ("z_2_alpha_1", 2.0, 1.0),
                                ("z_2_alpha_2", 2.0, 2.0), ("z_3_alpha_1", 3.0, 1.0),
                                ("z_3_alpha_2", 3.0, 2.0)]:
                d[key] = jet_angularity(xs, ys, zs, ws, eta_b, phi_b, axis, kappa=kap, alpha=alp, R=1.0)
        run("shape", shape)
    
        def signal_comp():
            d["zeroth_moment"] = zeroth_moment(i, j)
            r = return_EM_total_HAD_total(i, j)
            if r is not None:
                em, had = r
                tot = d["zeroth_moment"]
                d["EM_fraction"], d["HAD_fraction"] = em / tot, had / tot
            r = return_max_hit(i, j)
            if r is not None:
                tot = d["zeroth_moment"]
                d["most_energetic_hit_all"] = r[0] / tot
                d["most_energetic_hit_em"] = r[1] / tot
                d["most_energetic_hit_had"] = r[2] / tot
            d["core_signal_frac_all"] = core_signal_fraction(ws, id_cell, locs)
            (d["core_signal"], d["hcal_ratio"], d["hcal_over_all"], d["ecal_over_all"],
            d["hcal_over_hcal"], d["ecal_over_ecal"]) = core_signal_fraction_detector(ws, id_cell, locs)
        run("signal_comp", signal_comp)
    
        def timing():
            r = return_time_median_enertime_median(i, j)
            if r is None:
                return
            tc, ewc, ts, es = r
            d["time_center"], d["time_energyweighted_center"] = tc, ewc
            for deg, nm in [(2, "variance"), (3, "asymm"), (4, "tails")]:
                m, m_sr = return_time_moments(ts, es, ewc, deg)
                d[f"time_moment{deg}_{nm}"] = m
                d[f"time_moment{deg}_{nm}_sr"] = m_sr
            d["zeroth_moment_time"] = zeroth_moment_time(i, j)
            rt = return_barycenter_centre_of_time(i, j)
            if rt is not None:
                d["bary_eta_time"], d["bary_phi_time"], (d["cog_x_time"], d["cog_y_time"], d["cog_z_time"]) = rt[0], rt[1], rt[2]
        run("timing", timing)
    
        def isolations():
            axis, sL2, sT2, sLsq, sTsq, th_rms, mom_ratio, aspect = get_shower_axis(xs, ys, zs, ws, cog)
            d["isolation_all"] = isolation(unclaimed, j, i, ws, id_cell, locs, some_width)
            d["isolation_hits"] = isolation_allhits(j, i, ws, id_cell, locs, some_width)
            d["isolation_angular"] = isolation_eta_phi(unclaimed, j, i, ws, id_cell, locs,
                                                    angular_est, eta_b, phi_b)
            d["isolation_hits_angular"] = isolation_allhits_etaphi(j, i, ws, id_cell, locs,
                                                                angular_est, eta_b, phi_b)
            d["isolation_hits_but_all"], d["isolation_hits_but_clus"] = isolation_allhits_but(unclaimed, j, i, ws, id_cell, locs, sTsq / 2)
            #(unclaimed, cluster_idx, event_index, ws, cell_ids, locs, some_width)
        run("isolation", isolations)
    
        if errors:
            d["errors"] = " | ".join(errors)
        return d
    
    
    def dump_first_events(events, particle_info, n_events=20, angular_est=0.15,
                        csv_name=None, width=120):
        some_width = particle_info["test_width"]
        n_events = min(n_events, events.num_entries)
        rows = []
    
        for i in range(n_events):
            n_clusters = len(cluster_energy[i])
            unclaimed = unclaimed_indexes(i, n_clusters)
            entry_map = build_entry_point_map(i)
            for j in range(n_clusters):
                d = compute_cluster_variables(i, j, unclaimed, entry_map.get(j), some_width, angular_est)
                if d is None:
                    rows.append({"event": i, "cluster": j, "skipped": True})
                else:
                    rows.append({"event": i, "cluster": j, **d})
    
        # ---- final summary ----
        print("\n" + "=" * width)
        print(f"{particle_info['name']}: {n_events} events, {len(rows)} clusters")
        print("=" * width)
        for r in rows:
            print(f"--- event {r['event']}, cluster {r['cluster']} ---")
            if r.get("skipped"):
                print("skipped (no usable hits)\n")
                continue
            for k, v in r.items():
                if k in ("event", "cluster"):
                    continue
                print(f"{k}: {_fmt(v)}")
            print()
    
        if csv_name is None:
            csv_name = f"first_{n_events}_events_{particle_info['name']}.csv"
        pd.DataFrame(rows).to_csv(csv_name, index=False)
        print(f"Saved table to {csv_name}")
        return rows
    rows = dump_first_events(events, particle_info, n_events=20)
    continue   # skip the heavy full-dataset loops below while debugging
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

        #M is most, m is mean 
            
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


    #Let's have a plotting function
    #Let's just make a basic rudimentary plot for all of these, bar plot counts vs whatever variable this is

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
