import uproot
import numpy as np
import awkward as ak

FIELD_SPEC = [
    ("system", 5),
    ("side",   -2),   # signed
    ("layer",  6),
    ("module", 11),
    ("sensor", 8),
]

def decode_cellid(cellid, field_spec=FIELD_SPEC):
    cellid = int(cellid)
    result = {}
    offset = 0
    for name, width in field_spec:
        signed = width < 0
        w = abs(width)
        mask = (1 << w) - 1
        raw = (cellid >> offset) & mask
        if signed and (raw & (1 << (w - 1))):
            raw -= (1 << w)
        result[name] = raw
        offset += w
    return result

# --- point this at one file you already use ---
file_path = "/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root"

file = uproot.open(file_path)
events = file["events"]

pandora_clusters = events["PandoraClusters"]
pandora_clusters_hits = events["_PandoraClusters_hits"]
hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
hit_index_all  = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()

# grab the ECAL barrel cellID + position branches directly
ecal_barrel_cellid = events["EcalBarrelCollectionRec/EcalBarrelCollectionRec.cellID"].array()
ecal_barrel_x = events["EcalBarrelCollectionRec/EcalBarrelCollectionRec.position.x"].array()
ecal_barrel_y = events["EcalBarrelCollectionRec/EcalBarrelCollectionRec.position.y"].array()

ecal_endcap_cellid = events["EcalEndcapCollectionRec/EcalEndcapCollectionRec.cellID"].array()
ecal_endcap_x = events["EcalEndcapCollectionRec/EcalEndcapCollectionRec.position.x"].array()
ecal_endcap_y = events["EcalEndcapCollectionRec/EcalEndcapCollectionRec.position.y"].array()


hcal_barrel_cellid = events["HcalBarrelCollectionRec/HcalBarrelCollectionRec.cellID"].array()
hcal_barrel_x = events["HcalBarrelCollectionRec/HcalBarrelCollectionRec.position.x"].array()
hcal_barrel_y = events["HcalBarrelCollectionRec/HcalBarrelCollectionRec.position.y"].array()

ids = ak.values_astype(ecal_barrel_cellid, np.uint64)
system = ids & np.uint64(0x1F)
module = (ids >> 7) & np.uint64(0xFF)
stave = (ids >> 15) & np.uint(0x0F)
layer = (ids >> 19) & np.uint64(0x1FF)
system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}

# --- pick event 0, cluster 0 just to look ---
event_i = 0
cluster_j = 0

start = hits_begin_all[event_i][cluster_j]
end   = hits_end_all[event_i][cluster_j]
indices = hit_index_all[event_i][start:end]
ids     = collectionID_all[event_i][start:end]

print(f"Cluster has {end-start} hits total")

layers = []
rhos = []
n_shown = 0
for idx, sysid in zip(indices, ids):
    sysname = system2name.get(sysid, "UNKNOWN")
    if sysname != "EcalBarrelCollectionRec":
        continue  # only decode ECAL barrel cellIDs for this check
    layer_here = layer[event_i][idx]
    cellid = ecal_barrel_cellid[event_i][idx]
    x = ecal_barrel_x[event_i][idx]
    y = ecal_barrel_y[event_i][idx]
    rho = (x**2 + y**2) ** 0.5
    decoded = decode_cellid(cellid)
    print(f"hit idx={idx}  layer={decoded['layer']}  module={decoded['module']}  "
          f"sensor={decoded['sensor']}  rho={rho:.1f} mm, actuallayer={layer_here}")
    layers.append(decoded["layer"])
    rhos.append(rho)
    n_shown += 1

print(f"Entering Endcap")
for idx, sysid in zip(indices, ids):
    sysname = system2name.get(sysid, "UNKNOWN")
    if sysname != "EcalEndcapCollectionRec":
        continue  # only decode ECAL barrel cellIDs for this check
    layer_here = layer[event_i][idx]
    cellid = ecal_barrel_cellid[event_i][idx]
    x = ecal_barrel_x[event_i][idx]
    y = ecal_barrel_y[event_i][idx]
    rho = (x**2 + y**2) ** 0.5
    decoded = decode_cellid(cellid)
    print(f"hit idx={idx}  layer={decoded['layer']}  module={decoded['module']}  "
          f"sensor={decoded['sensor']}  rho={rho:.1f} mm, actuallayer={layer_here}")
    layers.append(decoded["layer"])
    rhos.append(rho)
    n_shown += 1
print("Entering HCAl")
for idx, sysid in zip(indices, ids):
    sysname = system2name.get(sysid, "UNKNOWN")
    if sysname != "HcalBarrelCollectionRec":
        continue  # only decode ECAL barrel cellIDs for this check
    layer_here = layer[event_i][idx]
    cellid = ecal_barrel_cellid[event_i][idx]
    x = ecal_barrel_x[event_i][idx]
    y = ecal_barrel_y[event_i][idx]
    rho = (x**2 + y**2) ** 0.5
    decoded = decode_cellid(cellid)
    print(f"hit idx={idx}  layer={decoded['layer']}  module={decoded['module']}  "
          f"sensor={decoded['sensor']}  rho={rho:.1f} mm, actuallayer={layer_here}")
    layers.append(decoded["layer"])
    rhos.append(rho)
    n_shown += 1

if n_shown == 0:
    print("No ECAL barrel hits found in this cluster — try a different event/cluster index.")
else:
    layers = np.array(layers)
    rhos = np.array(rhos)
    corr = np.corrcoef(layers, rhos)[0, 1]
    print(f"\nTotal ECAL barrel hits checked: {n_shown}")
    print(f"Correlation(layer, rho) = {corr:.3f}")
    print("  -> should be strongly positive (close to +1) if layer decode is correct,")
    print("     since higher layer = further out radially in a barrel ECAL.")

    # also show average rho per unique layer value, should increase monotonically
    print("\nMean rho per decoded layer:")
    for lay in sorted(set(layers)):
        mean_rho = rhos[layers == lay].mean()
        count = (layers == lay).sum()
        print(f"  layer={lay:3d}  mean_rho={mean_rho:8.1f} mm  (n={count})")