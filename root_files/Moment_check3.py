#This is my isolation check

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

some_width=0
def isolation_allhits_special(cluster_idx, event_index, ws, cell_ids, locs, some_width, eta_center, phi_center, max_dist):
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
    our_check = 0
    not_check = 0
    total_length_our = 0
    total_length_not = 0
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

        own_mask = np.isin(layer_global_idx, np.fromiter(own_idx_by_sys[sysname], dtype=int))
        our_idx = layer_global_idx[own_mask]
        candidate_idx = layer_global_idx[~own_mask]

        #I want to do all indices, and then colour code the ones within the cluster differently
        #Ah maybe keep two lists

        if len(candidate_idx) == 0:
            continue
        
        pos_x_ourclus = pos_x[our_idx]
        pos_y_ourclus = pos_y[our_idx]
        pos_z_ourclus = pos_z[our_idx]

        pos_x_notclus = pos_x[candidate_idx]
        pos_y_notclus = pos_y[candidate_idx]
        pos_z_notclus = pos_z[candidate_idx]

        phi_ourclus = np.arctan2(pos_y_ourclus, pos_x_ourclus)
        r_ourclus = np.sqrt(pos_x_ourclus**2 + pos_y_ourclus**2)
        theta_ourclus = np.arctan2(r_ourclus, pos_z_ourclus)
        eta_ourclus = -np.log(np.tan(theta_ourclus / 2))

        total_length_our += len(phi_ourclus)
        phi_notclus = np.arctan2(pos_y_notclus, pos_x_notclus)
        r_notclus = np.sqrt(pos_x_notclus**2 + pos_y_notclus**2)
        theta_notclus = np.arctan2(r_notclus, pos_z_notclus)
        eta_notclus = -np.log(np.tan(theta_notclus / 2))

        total_length_not += len(phi_notclus)
        delta_eta_our = eta_center - eta_ourclus
        delta_eta_not = eta_center - eta_notclus
        dphi_our = phi_center - phi_ourclus
        dphi_not = phi_center - phi_notclus
        delta_phi_our = np.arctan2(np.sin(dphi_our), np.cos(dphi_our))
        delta_phi_not = np.arctan2(np.sin(dphi_not), np.cos(dphi_not))
        r_our = np.sqrt(dphi_our**2 + delta_eta_our**2)
        r_not = np.sqrt(dphi_not**2 + delta_eta_not**2)
        our_check += np.sum(r_our <= max_dist)
        not_check += np.sum(r_not <= max_dist)
    
    if total_length_our >= 1:
        our_frac = our_check / total_length_our
    else:
        our_frac = 0
    if total_length_not >=1:
        not_frac = not_check / total_length_not
    else:
        not_frac = 0
    return our_frac, not_frac 


#Let's do this via eta and phi, also weight by layer, but radially







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


    def stats(lst):
        return (np.mean(lst), min(lst), max(lst)) if lst else (np.nan, np.nan, np.nan)

    dist_check_array = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]
    our_clus_median = []
    not_ourclus_median = []
    for max_dist in dist_check_array: 
        our_clus = []
        not_ourclus = []
        for event_index in range(events.num_entries):
            n_clusters = len(cluster_energy[event_index])
            #unclaimed = unclaimed_indexes(event_index, n_clusters)
            if n_clusters != 1:
                continue
            for cluster_index in range(n_clusters):
                result = return_barycenter_centre_of_gravity(event_index, cluster_index)
                if result is None:
                    continue
                eta_bary, phi_bary, _, _, _, _, ws, id_cell, locs = result
                our_hitnum, notclus_hitnum = isolation_allhits_special(cluster_index, event_index, ws, id_cell, locs, some_width, eta_bary, phi_bary, max_dist)
                #should be zero value
                our_clus.append(our_hitnum)
                not_ourclus.append(notclus_hitnum)
            #Do the full event
            #Also do the distribution at each event
            # Also do the distribution at each event here
        plt.figure()
        plt.hist(our_clus, bins=np.linspace(0, 1, 41), alpha=0.5, label="Cluster")
        plt.hist(not_ourclus, bins=np.linspace(0, 1, 41), alpha=0.5, label="Outside of Cluster")

        plt.xlabel("Fraction of hits")
        plt.ylabel("Number of events")
        plt.legend()
        plt.title(f"Hit fraction distribution, ΔR = {max_dist}, {particle_info['name']}")
        plt.savefig(f"{particle_info['name']}_{max_dist}.png")
        plt.close()
        our_clus_median.append(np.median(our_clus))
        not_ourclus_median.append(np.median(not_ourclus))
    x = np.arange(len(dist_check_array))
    width = 0.4

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width/2,
        our_clus_median,
        width,
        label="Hits in cluster"
    )

    plt.bar(
        x + width/2,
        not_ourclus_median,
        width,
        label="Hits outside cluster"
    )

    plt.xlabel(r"$\Delta R$ maximum")
    plt.ylabel("Fraction of hits")
    plt.title("Fraction of Hits Within Increasing $\Delta R$")

    plt.xticks(x, dist_check_array)
    plt.ylim(0, 1)

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        f"isolation_fraction_{particle_info['name']}.png",
        dpi=150
    )
    plt.close()

    #Now let's make a graph for pion and electrons


