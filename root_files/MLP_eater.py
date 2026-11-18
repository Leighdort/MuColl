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
import pandas as pd

import uproot
import awkward as ak

import matplotlib.pyplot as plt

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

####CONFIG##############################
SEED = 42
TRAIN_FRAC = 0.8
USE_CACHE = False          # True: skip extraction and load features_cache.npz
CACHE_FILE = "features_cache.npz"

# Keys stored per particle type (both in the cache and in the pooled data)
DATA_KEYS = ["X", "response", "mc_energy", "event_ids", "file_idx", "theta"]

FEATURE_NAMES = [
    "clus_iso", "all_iso", "iso_fraction", "cog_z", "radial_first",
    "aspect_ratio", "radial_rms", "time_center", "HAD_fraction", "max_hit_had",
    "core_signal_frac", "hcal_over_all", "ecal_over_all", "z_3_alpha_1",
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

base_path = "/users/rldohert/work/mucoll/mucoll-slurm/files/features_per_file1"
PREFIX = {"one_clus": "features", "max_iso": "features1", "min_iso": "features2"}
TAIL = "pt_10_200_theta_45-135"

def feature_files(pdg, choice):
    p = PREFIX[choice]
    if pdg == 11:
        return [f"{base_path}/{p}_pdg11_pdg_11_{TAIL}_{c}__reco_pdg_11_{TAIL}_nobib.npz"
                for c in ("calib", "calib2")]
    return [f"{base_path}/{p}_pdg211_pdg_211_{TAIL}_{c}__reco_pdg_211_{TAIL}_nobib_part{i}.npz"
            for c in ("calib2", "calib3") for i in (1, 2, 3)]

KEYS = ["X", "response", "mc_energy", "theta", "event_ids", "cluster_ids",
        "tag", "calib_prompt", "event_prompt_sum", "file_idx"]

def load_choice(choice, tags=None):
    out = {}
    for pdg in (11, 211):
        parts = []
        for f in feature_files(pdg, choice):
            with np.load(f) as z:
                parts.append({k: z[k] for k in KEYS})
        d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}   # X is 2D, still fine
        if tags is not None:
            m = np.isin(d["tag"], tags)
            d = {k: v[m] for k, v in d.items()}
        out[pdg] = d
    return out

all_particle_data = load_choice("min_iso")
print("min_iso")
######################### Train / test #########################
rng = np.random.default_rng(SEED)

n_per_class = min(len(d["response"]) for d in all_particle_data.values())
print(f"Using {n_per_class} clusters per particle type")

X_all, y_all, cls_all, E_all, Theta_all = [], [], [], [], []
for pdg, d in all_particle_data.items():
    idx = rng.permutation(len(d["response"]))[:n_per_class]
    X_all.append(d["X"][idx])
    y_all.append(d["response"][idx])
    E_all.append(d["mc_energy"][idx])
    Theta_all.append(d["theta"][idx])
    cls_all.append(np.full(n_per_class, 0 if pdg == 11 else 1))

X_all, y_all = np.vstack(X_all), np.concatenate(y_all)
E_all, cls_all = np.concatenate(E_all), np.concatenate(cls_all)
Theta_all = np.concatenate(Theta_all)          # fixed: was left as a list

idx_train, idx_test = train_test_split(
    np.arange(len(y_all)), train_size=TRAIN_FRAC,
    random_state=SEED, stratify=cls_all)

X_train, y_train, y_class_train = X_all[idx_train], y_all[idx_train], cls_all[idx_train]
X_test,  y_test,  y_class_test  = X_all[idx_test],  y_all[idx_test],  cls_all[idx_test]
E_test = E_all[idx_test]                       # truth energy: used for binning only, NOT a feature
Theta_test = Theta_all[idx_test]               # truth angle [deg]: used for binning only, NOT a feature
test_pdg = np.where(y_class_test == 1, 211, 11)

print("Training samples:", len(X_train))
print("Electron samples:", np.sum(y_class_train == 0))
print("Pion samples:", np.sum(y_class_train == 1))

# Scale features using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ################### MODEL ##############################

class MultiTaskMLP(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.regression = nn.Linear(32, 1)       # response
        self.classification = nn.Linear(32, 1)   # electron (0) vs pion (1)

    def forward(self, x):
        shared_features = self.shared(x)
        response = self.regression(shared_features)
        particle_logits = self.classification(shared_features)
        return response, particle_logits

# Split training data into training and validation
X_train_final, X_val, y_train_final, y_val, \
y_class_train_final, y_class_val = train_test_split(
    X_train_scaled,
    y_train,
    y_class_train,
    test_size=0.15,
    random_state=SEED,
    stratify=y_class_train
)

X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_final.reshape(-1, 1), dtype=torch.float32)
y_class_train_tensor = torch.tensor(y_class_train_final.reshape(-1, 1), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
y_class_val_tensor = torch.tensor(y_class_val.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

#################### MULTI-TASK TRAINING ##############################

torch.manual_seed(4)

model = MultiTaskMLP(input_size=X_train_scaled.shape[1])

regression_loss = nn.MSELoss()
classification_loss = nn.BCEWithLogitsLoss()  # 1 = pion

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

classification_weight = 0.5

N_EPOCHS = 500
patience = 20
best_val_loss = float("inf")
epochs_without_improvement = 0

train_losses, val_losses = [], []
train_reg_losses, val_reg_losses = [], []
train_class_losses, val_class_losses = [], []

best_model_state = None

dataset = TensorDataset(X_train_tensor, y_train_tensor, y_class_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(N_EPOCHS):

    model.train()

    epoch_total = 0
    epoch_reg = 0
    epoch_class = 0

    for X_batch, y_batch, class_batch in loader:

        optimizer.zero_grad()

        response_pred, particle_logits = model(X_batch)

        reg_loss = regression_loss(response_pred, y_batch)
        class_loss = classification_loss(particle_logits, class_batch)

        total_loss = reg_loss + classification_weight * class_loss

        total_loss.backward()
        optimizer.step()

        epoch_total += total_loss.item()
        epoch_reg += reg_loss.item()
        epoch_class += class_loss.item()

    train_total = epoch_total / len(loader)
    train_reg = epoch_reg / len(loader)
    train_class = epoch_class / len(loader)

    train_losses.append(train_total)
    train_reg_losses.append(train_reg)
    train_class_losses.append(train_class)

    model.eval()

    with torch.no_grad():
        val_response, val_logits = model(X_val_tensor)
        val_reg = regression_loss(val_response, y_val_tensor).item()
        val_class = classification_loss(val_logits, y_class_val_tensor).item()
        val_total = val_reg + classification_weight * val_class

    val_losses.append(val_total)
    val_reg_losses.append(val_reg)
    val_class_losses.append(val_class)

    # Early stopping: monitor validation regression MSE
    if val_reg < best_val_loss - 1e-5:
        best_val_loss = val_reg
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        epochs_without_improvement += 1

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train Reg: {train_reg:.6f} | "
            f"Val Reg: {val_reg:.6f} | "
            f"Train Class: {train_class:.6f} | "
            f"Val Class: {val_class:.6f}"
        )

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_model_state)

print("\nTraining complete")
print("Best validation regression MSE:", best_val_loss)

# ################### TEST PREDICTIONS ##############################

model.eval()

with torch.no_grad():
    predictions_tensor, classification_tensor = model(X_test_tensor)
    predictions = predictions_tensor.numpy().flatten()
    classification_probabilities = torch.sigmoid(classification_tensor).numpy().flatten()

classification_predictions = (classification_probabilities >= 0.5).astype(int)

# ################### CLASSIFICATION RESULTS ##############################

print("\nClassification results")
print("Accuracy:", accuracy_score(y_class_test, classification_predictions))
print("AUC:", roc_auc_score(y_class_test, classification_probabilities))

print("\nClassification report:")
print(classification_report(y_class_test, classification_predictions,
                            target_names=["Electron", "Pion"]))

print("\nConfusion matrix:")
print(confusion_matrix(y_class_test, classification_predictions))

# ################### TRAINING LOSS PLOT ##############################
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_reg_losses, label="Training Regression")
ax.plot(val_reg_losses, label="Validation Regression")
ax.plot(train_class_losses, label="Training Classification")
ax.plot(val_class_losses, label="Validation Classification")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale("log")
ax.set_title("Multi-Task MLP Training")
ax.legend()
fig.tight_layout()
fig.savefig("mlp_multitask_loss_both2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ################### ACTUAL VS PREDICTED: OVERALL ##############################

lo = min(y_test.min(), predictions.min())
hi = max(y_test.max(), predictions.max())
bins = np.linspace(lo, hi, 40)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(y_test, bins=bins, histtype="stepfilled", alpha=0.45, color="royalblue",
        label=f"Actual (mean={y_test.mean():.3f}, σ={y_test.std():.3f})")
ax.hist(predictions, bins=bins, histtype="stepfilled", alpha=0.45, color="darkorange",
        label=f"Predicted (mean={predictions.mean():.3f}, σ={predictions.std():.3f})")
ax.set_xlabel("Response (E_cluster / E_true)")
ax.set_ylabel("Clusters per bin")
ax.set_title("Both Test set: Actual vs. Predicted Response")
ax.legend()
fig.tight_layout()
fig.savefig("mlp_actual_vs_pred_hist_both2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ################### PERFORMANCE / PLOTS BY PARTICLE AND TRUTH ENERGY ##############################

# ################### ACTUAL VS PREDICTED: OVERALL ##############################
# NOTE: this block is a duplicate of the one above and overwrites the same file.
# Delete one of them if you only want a single version.

lo = min(y_test.min(), predictions.min())
hi = max(y_test.max(), predictions.max())
bins = np.linspace(lo, hi, 40)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(y_test, bins=bins, histtype="stepfilled", alpha=0.45, color="gray",
        label=f"Actual (mean={y_test.mean():.3f}, σ={y_test.std():.3f})")
ax.hist(predictions, bins=bins, histtype="stepfilled", alpha=0.45, color="purple",
        label=f"Predicted (mean={predictions.mean():.3f}, σ={predictions.std():.3f})")
ax.set_xlabel("Response (E_cluster / E_true)")
ax.set_ylabel("Clusters per bin")
ax.set_title("Both Test set: Actual vs. Predicted Response")
ax.legend()
fig.tight_layout()
fig.savefig("mlp_actual_vs_pred_hist_both2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ################### PERFORMANCE / PLOTS BY PARTICLE AND TRUTH ENERGY ##############################

particle_types = {211: "Pions", 11: "Electrons"}

# light = actual / uncalibrated, dark = predicted / calibrated
particle_colors = {
    11:  dict(light="deepskyblue", dark="navy"),
    211: dict(light="orange",      dark="firebrick"),
}

def make_plots(pdg, tag, title, mask):
    actual, predicted = y_test[mask], predictions[mask]
    if len(actual) < 20:
        print(f"Skipping {title}: only {len(actual)} clusters")
        return
    name = particle_types[pdg]
    light, dark = particle_colors[pdg]["light"], particle_colors[pdg]["dark"]
    calibration = actual / predicted
    ratio = predicted / actual

    print(f"{name} {title}: N = {len(actual)}, "
          f"R² = {r2_score(actual, predicted):.4f}, "
          f"MSE = {mean_squared_error(actual, predicted):.6f}, "
          f"mean actual = {actual.mean():.4f}, mean predicted = {predicted.mean():.4f}")

    def hist(arrs, labels, colors, xlabel, ttl, fname):
        lo = min(a.min() for a in arrs); hi = max(a.max() for a in arrs)
        b = np.linspace(lo, hi, 40)
        fig, ax = plt.subplots(figsize=(7, 5))
        for a, l, c in zip(arrs, labels, colors):
            ax.hist(a, bins=b, histtype="stepfilled", alpha=0.45, color=c,
                    label=f"{l} (mean={a.mean():.3f}, σ={a.std():.3f})")
        ax.set_xlabel(xlabel); ax.set_ylabel("Clusters per bin")
        ax.set_title(ttl); ax.legend()
        fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    hist([actual, predicted], ["Actual", "Predicted"], [light, dark],
         "Response (E_cluster / E_true)", f"{name}, {title}: Actual vs. Predicted",
         f"mlp_actual_vs_pred_{pdg}_{tag}2.png")
    hist([ratio], ["Stats"], [dark],
         "Predicted / Actual", f"{name}, {title}: Predicted / Actual",
         f"mlp_predicted_actual_{pdg}_{tag}2.png")
    hist([actual, calibration], ["Uncalibrated", "Calibrated"], [light, dark],
         "Response", f"{name}, {title}: Calibrated & Uncalibrated",
         f"mlp_calibrated_{pdg}_{tag}2.png")

print("\nPerformance by particle and truth energy:")

# fixed edges; or use np.quantile(E_test, np.linspace(0, 1, 6)) for equal statistics per bin
E_edges = np.linspace(10, 290, 15)
for pdg in (11, 211):
    for lo, hi in zip(E_edges[:-1], E_edges[1:]):
        m = (test_pdg == pdg) & (E_test >= lo) & (E_test < hi)
        make_plots(pdg, f"E{lo:.0f}-{hi:.0f}GeV", f"{lo:.0f}-{hi:.0f} GeV", m)

# ##########################################################################################  #
########################   Summery Graphs    #############################
QLO, QHI = 16, 84
MIN_N = 5

styles = {
    (11,  "unc"): dict(color="deepskyblue", marker="o", dx=-0.15),
    (11,  "cal"): dict(color="navy",        marker="s", dx=-0.05),
    (211, "unc"): dict(color="orange",      marker="o", dx= 0.05),
    (211, "cal"): dict(color="firebrick",   marker="s", dx= 0.15),
}
labels = {"unc": "Uncalibrated", "cal": "Calibrated"}

def bin_stats(values, x, pdg_mask, edges):
    med, lo, hi, res = [], [], [], []
    last = len(edges) - 2
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        # include the upper edge in the last bin
        upper = (x <= b) if i == last else (x < b)
        m = pdg_mask & (x >= a) & upper
        if m.sum() < MIN_N:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); res.append(np.nan)
            continue
        q_lo, q_med, q_hi = np.percentile(values[m], [QLO, 50, QHI])
        med.append(q_med); lo.append(q_lo); hi.append(q_hi)
        res.append(((q_hi - q_lo) / 2) / q_med)
    return np.array(med), np.array(lo), np.array(hi), np.array(res)

calibrated_all = y_test / predictions

def summary_plots(x, edges, xlabel, tag):
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    stats = {}
    for pdg in (11, 211):
        pmask = (test_pdg == pdg)
        stats[(pdg, "unc")] = bin_stats(y_test,         x, pmask, edges)
        stats[(pdg, "cal")] = bin_stats(calibrated_all, x, pmask, edges)

    # ---- Response ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (pdg, kind), s in styles.items():
        med, lo, hi, _ = stats[(pdg, kind)]
        ax.errorbar(centers + s["dx"] * width, med,
                    yerr=[med - lo, hi - med],
                    fmt=s["marker"], capsize=3, color=s["color"],
                    label=f"{labels[kind]}, {particle_types[pdg]}")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Response (median, 16–84% band)")
    ax.set_title(f"Response vs. {xlabel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"mlp_response_vs_{tag}2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Resolution ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (pdg, kind), s in styles.items():
        ax.plot(centers, stats[(pdg, kind)][3], marker=s["marker"], ls="-",
                color=s["color"], label=f"{labels[kind]}, {particle_types[pdg]}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Resolution  $\sigma/\mathrm{median}$")
    ax.set_title(f"Resolution vs. {xlabel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"mlp_resolution_vs_{tag}2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# Energy (20 GeV bins)
summary_plots(E_test, np.arange(10, 211, 20), "True Energy [GeV]", "E")

# Angle (10 degree bins, 45-135 to match the generated theta range)
summary_plots(Theta_test, np.arange(45, 136, 10), "True Angle [deg]", "angle")   # fixed: was Angle_test





# ######################## Per-particle summary: energy on x, one line per angle slice ########################

# ######################## Per-particle summary: energy on x, one line per angle slice (unc + cal) ########################
from matplotlib.lines import Line2D

# Angle chunks (edit these to taste); each one becomes a color on the plot
ANGLE_EDGES = [45, 65, 85, 105, 125, 135]
E_EDGES_ANGLE = np.arange(10, 211, 20)

# Style per calibration state: open/dashed = uncalibrated, filled/solid = calibrated
KIND_STYLE = {
    "unc": dict(marker="o", ls="--", filled=False, label="Uncalibrated"),
    "cal": dict(marker="s", ls="-",  filled=True,  label="Calibrated"),
}

def energy_by_angle_plots(pdg):
    name = particle_types[pdg]
    pmask = (test_pdg == pdg)
    centers = 0.5 * (E_EDGES_ANGLE[:-1] + E_EDGES_ANGLE[1:])
    width = E_EDGES_ANGLE[1] - E_EDGES_ANGLE[0]

    n_ang = len(ANGLE_EDGES) - 1
    n_series = 2 * n_ang
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_ang))
    values = {"unc": y_test, "cal": calibrated_all}

    # stats[(angle_index, kind)] = (med, lo, hi, res)
    stats = {}
    for i in range(n_ang):
        a, b = ANGLE_EDGES[i], ANGLE_EDGES[i + 1]
        upper = (Theta_test <= b) if i == n_ang - 1 else (Theta_test < b)
        amask = pmask & (Theta_test >= a) & upper
        for kind in KIND_STYLE:
            stats[(i, kind)] = bin_stats(values[kind], E_test, amask, E_EDGES_ANGLE)

    # Shared legend handles: colors = angle, markers = calibration
    angle_handles = [Line2D([0], [0], color=cmap[i], lw=2,
                            label=f"{ANGLE_EDGES[i]}–{ANGLE_EDGES[i+1]}°")
                     for i in range(n_ang)]
    kind_handles = [Line2D([0], [0], color="k", marker=s["marker"], ls=s["ls"],
                           mfc="k" if s["filled"] else "none", label=s["label"])
                    for s in KIND_STYLE.values()]

    def finish(ax, ylabel, title, fname):
        ax.set_xlabel("True Energy [GeV]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        leg1 = ax.legend(handles=angle_handles, title="Theta", loc="upper right")
        ax.add_artist(leg1)
        ax.legend(handles=kind_handles, loc="lower right")
        ax.figure.tight_layout()
        ax.figure.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

    # ---- Response ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_ang):
        for j, (kind, s) in enumerate(KIND_STYLE.items()):
            med, lo, hi, _ = stats[(i, kind)]
            k = 2 * i + j
            dx = (k - (n_series - 1) / 2) / n_series * 0.7 * width   # offset so bars don't overlap
            ax.errorbar(centers + dx, med, yerr=[med - lo, hi - med],
                        fmt=s["marker"], capsize=2, color=cmap[i],
                        mfc=cmap[i] if s["filled"] else "none")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    finish(ax, "Response (median, 16–84% band)",
           f"{name}: Response vs. Energy, by angle",
           f"mlp_response_vs_E_by_angle_{pdg}2.png")

    # ---- Resolution ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_ang):
        for kind, s in KIND_STYLE.items():
            res = stats[(i, kind)][3]
            ax.plot(centers, res, marker=s["marker"], ls=s["ls"], color=cmap[i],
                    mfc=cmap[i] if s["filled"] else "none")
    finish(ax, r"Resolution  $\sigma/\mathrm{median}$",
           f"{name}: Resolution vs. Energy, by angle",
           f"mlp_resolution_vs_E_by_angle_{pdg}2.png")

for pdg in (11, 211):
    energy_by_angle_plots(pdg)