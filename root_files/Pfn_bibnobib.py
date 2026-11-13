from __future__ import absolute_import, division, print_function
import numpy as np
import re
import ast
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import numpy as np
import tensorflow.keras

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from energyflow.archs.efn import PFN
from tensorflow.keras.callbacks import ReduceLROnPlateau



pid = 211
e = 50
a = 85


# System IDs
ECAL_BARREL = 679272617
HCAL_BARREL = 1573202488
ECAL_ENDCAP = 3383333369
HCAL_ENDCAP = 2381985645

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec"
}

valid_systems = tuple(system2name.keys())

print("Imports are Complete")

datasets = [
    {
        "matches": "/users/rldohert/work/mucoll/mucoll-slurm/files/matches_500f.txt",
        "root": f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_clustertest/reco_pdg_{pid}_pt_{e}_theta_{a}-{a}_bib.root"
    },
    {
        "matches": "/users/rldohert/work/mucoll/mucoll-slurm/files/matches_500_clustertest2.txt",
        "root": f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_clustertest2/reco_pdg_{pid}_pt_{e}_theta_{a}-{a}_bib.root"
    }
]
# ==========================
# Parse matches file
# ==========================

match_dict = {}
cluster_list_bib = []
for dataset in datasets:
    match_dict = {}
    current_event = None
    #Can I combine this dataset with matches_500_clustertest2.txt   
    #And use it with the same file but with clustertest2 next to it?
    with open(dataset["matches"]) as f:
        for line in f:
            line = line.strip()

            if line.startswith("Event:"):
                m = re.match(r"Event:\s*(\d+)(.*)", line)

                current_event = int(m.group(1))
                match_dict[current_event] = []

                remainder = m.group(2).strip()

                if remainder.startswith("("):
                    tup = ast.literal_eval(remainder)   # faster + safer
                    match_dict[current_event].append(tup)

            elif line.startswith("("):
                tup = ast.literal_eval(line)
                match_dict[current_event].append(tup)


    # ==========================
    # Load ROOT
    # ==========================

    pid = 211
    e = 50
    a = 85

    file = uproot.open(dataset["root"])
    events = file["events"]


    pch = events["_PandoraClusters_hits"]
    clusters = events["PandoraClusters"]

    hit_index_all    = pch["_PandoraClusters_hits.index"].array()
    collectionID_all = pch["_PandoraClusters_hits.collectionID"].array()
    hits_begin_all   = clusters["PandoraClusters.hits_begin"].array()
    hits_end_all     = clusters["PandoraClusters.hits_end"].array()


    # ==========================
    # Load detector hit arrays
    # ==========================

    posx, posy, posz, energy_hits, hit_time = {}, {}, {}, {}, {}

    for name in system2name.values():
        prefix = f"{name}/{name}"

        posx[name]       = events[f"{prefix}.position.x"].array()
        posy[name]       = events[f"{prefix}.position.y"].array()
        posz[name]       = events[f"{prefix}.position.z"].array()
        energy_hits[name]= events[f"{prefix}.energy"].array()
        hit_time[name]   = events[f"{prefix}.time"].array()


    # ==========================
    # Main loop
    # ==========================
    # =====================================================================
    # Optimized Vectorized Main Loop
    # =====================================================================
    for i in range(events.num_entries):
        event_matches = match_dict.get(i, [])
        
        # Get lengths of clusters in this event
        h_begin = hits_begin_all[i]
        h_end = hits_end_all[i]
        num_clusters = len(h_begin)
        if num_clusters == 0:
            continue

        hit_index = hit_index_all[i]
        collection_ID = collectionID_all[i]

        # Pre-cache all detector hit arrays for this specific event to numpy
        # Doing this conversion once per event is significantly faster!
        evt_hits = {}
        for coll_id, sys_name in system2name.items():
            evt_hits[coll_id] = {
                "x": ak.to_numpy(posx[sys_name][i]),
                "y": ak.to_numpy(posy[sys_name][i]),
                "z": ak.to_numpy(posz[sys_name][i]),
                "e": ak.to_numpy(energy_hits[sys_name][i]),
                "t": ak.to_numpy(hit_time[sys_name][i])
            }

        # Create base labels (1 = BIB)
        cluster_labels = {j: 1 for j in range(num_clusters)}
        for _, matched_id, _ in event_matches:
            cluster_labels[matched_id] = 0   # 0 = Non-BIB (Signal)

        if i % 100 == 0:
            print(f"Processing Event {i}/{events.num_entries} | Clusters: {num_clusters} | Matched: {len(event_matches)}")

        # Loop over clusters
        for j in range(num_clusters):
            start = h_begin[j]
            end = h_end[j]

            # Slice out hit indices and system IDs belonging to this cluster
            c_hit_idx = hit_index[start:end]
            c_coll_id = collection_ID[start:end]

            hit_features = []

            # Extract hits belonging to valid detector collections
            for coll_id in valid_systems:
                mask = (c_coll_id == coll_id)
                if not np.any(mask):
                    continue

                idxs = c_hit_idx[mask]
                sys_data = evt_hits[coll_id]

                # Extract features safely using pre-built numpy arrays
                feat = np.column_stack([
                    sys_data["x"][idxs],
                    sys_data["y"][idxs],
                    sys_data["z"][idxs],
                    sys_data["e"][idxs],
                    sys_data["t"][idxs]
                ])
                hit_features.append(feat)

            if hit_features:
                all_hits = np.concatenate(hit_features)
                cluster_list_bib.append({
                    "hits": all_hits,
                    "label": cluster_labels[j]
                })
    
    '''
    for i in range(events.num_entries):

        

        event_matches = match_dict[i]

        # faster label creation
        


        # event-local cache (FASTER than nested dict)
        event_x = {}
        event_y = {}
        event_z = {}
        event_e = {}
        event_t = {}

        for sys_name in system2name.values():
            event_x[sys_name] = posx[sys_name][i]
            event_y[sys_name] = posy[sys_name][i]
            event_z[sys_name] = posz[sys_name][i]
            event_e[sys_name] = energy_hits[sys_name][i]
            event_t[sys_name] = hit_time[sys_name][i]

        hits_begin_arr = hits_begin_all[i]
        hits_end_arr   = hits_end_all[i]
        hit_index      = hit_index_all[i]
        collection_ID  = collectionID_all[i]

        cluster_labels = {j: 1 for j in range(len(hits_begin_arr))}

        for cluster_idx, matched_id, score in event_matches:
            cluster_labels[matched_id] = 0   # non-BIB
        
        if i % 50 == 0:
            print(
                "event", i,
                "clusters", len(hits_begin_arr),
                "matched", len(event_matches)
            )
        # loop over clusters
        for j in range(len(hits_begin_arr)):

            start = hits_begin_arr[j]
            end   = hits_end_arr[j]

            cluster_hit_idx = hit_index[start:end]
            cluster_coll_id = collection_ID[start:end]

            hit_features = []

            # avoid np.unique() every cluster
            for coll_id in valid_systems:

                mask = (cluster_coll_id == coll_id)

                # skip if detector absent
                if not np.any(mask):
                    continue

                idxs = cluster_hit_idx[mask]

                sys = system2name[coll_id]

                feat = np.column_stack([
                    ak.to_numpy(event_x[sys][idxs]),
                    ak.to_numpy(event_y[sys][idxs]),
                    ak.to_numpy(event_z[sys][idxs]),
                    ak.to_numpy(event_e[sys][idxs]),
                    ak.to_numpy(event_t[sys][idxs])
                ])

                hit_features.append(feat)

            if hit_features:

                all_hits = np.concatenate(hit_features)
                label = cluster_labels.get(j, 1)
                cluster_list_bib.append({
                    "hits": all_hits,
                    "label": cluster_labels.get(j, 1)
                })
                if i == 0 and label == 0:
                    print(f"cluster {j} is NON-BIB (label=0), matched_id={event_matches[j][1] if j < len(event_matches) else None}")
    '''
    #Now we will try to do r = square root x^2 +  y^2 + z^2 
    #Then we will do x / r, y / r, z / y, e / sum all energy in the event, then t 
    #We will then need to find the cluster with the most number of hits and zero pad


#we want to fill cluster_list_bib2
cluster_list_bib2 = []

for cluster in cluster_list_bib:

    hits = cluster["hits"]

    x = hits[:,0]
    y = hits[:,1]
    z = hits[:,2]
    e = hits[:,3]
    t = hits[:,4]

    r = np.sqrt(x*x + y*y + z*z)

    transformed = np.column_stack([
        x / r,
        y / r,
        z / r,
        e,
        t
    ])

    cluster_list_bib2.append({
        "hits": transformed,
        "label": cluster["label"]
    })


#max_hits = max(cluster["hits"].shape[0] for cluster in cluster_list_bib2)
max_hits = 500
#We want to normalize here

all_unpadded_e = []
all_unpadded_t = []

for cluster in cluster_list_bib2:
    hits = cluster["hits"]
    all_unpadded_e.append(hits[:, 3])  # energy channel
    all_unpadded_t.append(hits[:, 4])  # time channel

# Flatten into single arrays
all_unpadded_e = np.concatenate(all_unpadded_e)
all_unpadded_t = np.concatenate(all_unpadded_t)

# Calculate global constants
epsilon = 1e-6
log_e_global = np.log10(all_unpadded_e + epsilon)

mu_e = np.mean(log_e_global)
sigma_e = np.std(log_e_global)

mu_t = np.mean(all_unpadded_t)
sigma_t = np.std(all_unpadded_t)




#Now we need to do padding:

def pad_hits(hits, max_hits):
    """
    hits: (N, 5)
    returns: (max_hits, 5)
    """
    n = hits.shape[0]
    
    #if n >= max_hits:
    #    return hits[:max_hits]

    padded = np.zeros((max_hits, 6), dtype=hits.dtype)
    if n > 0:
        actual_hits = hits[:max_hits]
        actual_n = actual_hits.shape[0]

        padded[:actual_n, :5] = actual_hits
        padded[:actual_n, 5] = 1.0
    #pad_value = 0
    #padded = np.full((max_hits, hits.shape[1]), pad_value, dtype=hits.dtype)
    #padded[:n] = hits
    return padded

for cluster in cluster_list_bib2:
    cluster["hits"][:, 3] = (np.log10(cluster["hits"][:, 3] + epsilon) - mu_e) / sigma_e
    cluster["hits"][:, 4] = (cluster["hits"][:, 4] - mu_t) / sigma_t
    cluster["hits"] = pad_hits(cluster["hits"], max_hits)
    

labels = [cluster["label"] for cluster in cluster_list_bib2]

n_zero = labels.count(0)
n_one = labels.count(1)

print("Label 0:", n_zero)
print("Label 1:", n_one)
print("Total clusters:", len(labels))




#We have 746 nobib clusters, 200k bib clusters
#Let's downgrade this to 5000 bib clusters (important to shuffle so they're not all from the same event)

nobib = [c for c in cluster_list_bib2 if c["label"] == 0]
bib = [c for c in cluster_list_bib2 if c["label"] == 1]
np.random.shuffle(bib)  # in-place shuffle
bib_reduced = bib[:30000] #first 5000 events of shuffle_bib

np.random.shuffle(nobib)
np.random.shuffle(bib_reduced)

#Now we want to do 
#We want 80% train, 20 % validation
#Also we may want to normalize all the numbers?
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
#Phi_sizes, F_sizes = (128, 128, 128, 256), (128, 128, 128)
Phi_sizes, F_sizes = (32, 32, 64), (64, 32)
num_epoch = 100
batch_size = 50

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

#Now we need to strip the labels off of nobib, strip hits from label
X_nobib = [cluster["hits"] for cluster in nobib]
Y_nobib = [cluster["label"] for cluster in nobib]
X_bib = [cluster["hits"] for cluster in bib_reduced]
Y_bib = [cluster["label"] for cluster in bib_reduced]

n_nobib = len(Y_nobib)
n_bib = len(Y_bib)
total = n_nobib + n_bib

weight_n_nobib = total / (2 * n_nobib)
weight_n_bib = total / (2 * n_bib)

class_weights = {
    0: weight_n_nobib,
    1: weight_n_bib
}
'''
(X_trainno, X_valno, X_testno, Y_trainno, Y_valno, Y_testno) = data_split(X_nobib, Y_nobib, train=0.8, val=0.1, test=0.1)
(X_trainbib, X_valbib, X_testbib, Y_trainbib, Y_valbib, Y_testbib) = data_split(X_bib, Y_bib, train=0.8, val=0.1, test=0.1)
:q!

X_train = X_trainno + X_trainbib
Y_train = Y_trainno + Y_trainbib
X_val = X_valno + X_valbib
Y_val = Y_valno + Y_valbib
X_test = X_testno + X_testbib
Y_test = Y_testno + Y_testbib

#Now we want to shuffle the trains, vals, and tests together 
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
X_val, Y_val = shuffle(X_val, Y_val, random_state = 43)
X_test, Y_test = shuffle(X_test, Y_test, random_state = 44)
'''

#Instead of standard test, train, val, I will be using k-folding
print(type(X_nobib))
print(type(X_bib))

print(type(X_nobib[0]))
print(type(X_bib[0]))

print(X_nobib[0].shape)
print(X_bib[0].shape)

#Is kfolds better for smaller datasets
X = X_nobib + X_bib
Y = Y_nobib + Y_bib
'''
X = pd.concat([X_nobib, X_bib], ignore_index = True)
Y = pd.concat([Y_nobib, Y_bib], ignore_index = True)
'''
'''
brem0 = pd.concat([brem0_lower, brem0_upper], ignore_index=True)
brem1 = pd.concat([brem1_lower, brem1_upper], ignore_index=True)
brem2 = pd.concat([brem2_lower, brem2_upper], ignore_index=True)
'''
X, Y = shuffle(X, Y, random_state =35)
kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int32)
#Y = tf.keras.utils.to_categorical(Y, num_classes=2)
lengths = []

for cluster in cluster_list_bib2:
    lengths.append(np.sum(cluster["hits"][:,5]))

print("Median:", np.median(lengths))
print("95th:", np.percentile(lengths,95))
print("99th:", np.percentile(lengths,99))
print("Max:", np.max(lengths))
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("unique labels:", np.unique(Y))

class BinarySoftmaxAUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[:, 1]
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)
out_of_fold_auc_fpr = []
out_of_fold_auc_tpr = []
for i, (train_idx, val_idx) in enumerate(kf.split(X, Y)):
    optimizer = Adam(learning_rate = 5e-5)
    #pfn = PFN(input_dim=6, Phi_sizes=Phi_sizes, F_sizes=F_sizes, optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name='auc')])
    pfn = PFN(input_dim=6, Phi_sizes=Phi_sizes, F_sizes=F_sizes, optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy", BinarySoftmaxAUC(name="auc")])
    es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=20)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )
    
    mc = ModelCheckpoint(f'best_pfn_fold_{i}.keras', monitor='val_auc', mode='max', verbose=1, save_best_only=True)

    train_X = X[train_idx]
    pred_check = pfn.predict(train_X[:5])
    print(pred_check.shape)
    print(pred_check)
    train_Y = Y[train_idx]
    val_X = X[val_idx]
    val_Y= Y[val_idx]
    
    history = pfn.fit(train_X, train_Y,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(val_X, val_Y),
        class_weight = class_weights,
        verbose=1,
        callbacks=[es,mc, lr_reducer],
    )
    preds = pfn.predict(val_X)
    pred_prob_bib = preds[:, 1] #probability that its bib
    preds_prob_bib_isbib = pred_prob_bib[val_Y == 1]
    preds_prob_bib_notbib = pred_prob_bib[val_Y == 0]

    #Now I am going to plot these two values 
    plt.hist(
            preds_prob_bib_notbib, 
            bins=50, 
            range=(0, 1), 
            alpha=0.5, 
            color='tab:blue', 
            label='True non-BIB (Signal)', 
            density=True
    )

        # 2. Plot True BIB (Background) - Should cluster tightly near 1.0
    plt.hist(
        preds_prob_bib_isbib, 
        bins=50, 
        range=(0, 1), 
        alpha=0.5, 
        color='tab:orange', 
        label='True BIB (Background)', 
        density=True
    )
    plt.xlabel('PFN Output Probability (Score = 1.0 is BIB)')
    plt.title(f"PFN Discriminator")
    plt.yscale('log')
    plt.savefig(f"PFN_Discriminator_Fold_{i+1}.pdf", bbox_inches='tight')
    plt.close()

    #Plot training & validation accuracy w/in folds
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"PFN Training/Validation Accuracy Fold {i}")
    plt.savefig(f"Accuracy_PFN_{i}_full.pdf")
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label=' Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"PFN Training/Validation Loss Fold {i}")
    plt.savefig(f"Loss_PFN_{i}_full.pdf")
    plt.close()

    #Then I want to do it based on AUC
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"PFN Loss Fold {i+1}")
    plt.legend()
    plt.savefig(f"loss_fold_{i+1}_full.pdf")
    plt.close()

    pred = pfn.predict(val_X)

    # probability for class 1
    pred_prob = pred[:,1]

    fpr, tpr, _ = roc_curve(val_Y, pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve Fold {i+1}")
    plt.legend()
    plt.savefig(f"roc_fold_{i+1}_full.pdf")
    plt.close()

#Now we are going to graph an oof auc

