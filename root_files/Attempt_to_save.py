
import uproot
import numpy as np
import pandas as pd
import pickle

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec"
}


#electron_energies = [1, 2]

#electron_events = {
#    1: [0, 50, 269, 2678],
#    2: [0, 100, 2663, 6595],
#}

electron_events = {
    5: [0,32,9400,9664,9748],
    10: [0,3071,4000,4536],
    50: [0,1527,2211,3116,9948],
    100: [0,436,500,1289,7401,9228],
    150: [0,370,2978,5690,9900],
    200: [980,2497,6919,9940]

}

pion_1=[0,64,2525,3506,4392,9950]
pion_2=[1,286,2075,6504,9930]
lister = list(range(5900, 6001))
pion_50=[0,1435,3422,3508,6015,6030,6874] #maybe 6015 normal
pion_50.extend(lister) #Use extend not append
pion_energies = [1, 2, 5, 10, 50, 100, 150, 200]
pion_events = {
    1: [0,64,2525,3506,4392,9950],
    2: [1,286,2075,6504,9930], 
    5: [0,3611,9900], 
    10: [0,1417,7290,9900], 
    50: pion_50,
    100: [1,446,8485], 
    150: [0,1,771,5531,5785,6288,9114],
    200: [0,2,1124,1434,2029]
}

def process_event(events, i, system2name):

    mcparticles = events["MCParticles"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    clusters = events["PandoraClusters"]
    pch = events["_PandoraClusters_hits"]

    particle_ids = mcparticles["MCParticles.PDG"].array()[i]

    px = mcparticles["MCParticles.momentum.x"].array()[i]
    py = mcparticles["MCParticles.momentum.y"].array()[i]
    pz = mcparticles["MCParticles.momentum.z"].array()[i]
    mass = mcparticles["MCParticles.mass"].array()[i]

    total_momentum = np.sqrt(px**2 + py**2 + pz**2)
    total_energy = np.sqrt(total_momentum**2 + mass**2)

    status_gen = mcparticles["MCParticles.generatorStatus"].array()[i]
    num_tracks = len(deduped_tracks["DedupedTracks_objIdx.index"].array()[i])

    hit_index_all    = pch["_PandoraClusters_hits.index"].array()
    collectionID_all = pch["_PandoraClusters_hits.collectionID"].array()
    hits_begin_all   = clusters["PandoraClusters.hits_begin"].array()
    hits_end_all     = clusters["PandoraClusters.hits_end"].array()

    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    hit_index      = hit_index_all[i]
    collection_ID  = collectionID_all[i]

    # Load hit positions
    posx, posy, posz, energy = {}, {}, {}, {}
    for name in system2name.values():
        prefix = f"{name}/{name}"
        posx[name] = events[f"{prefix}.position.x"].array()
        posy[name] = events[f"{prefix}.position.y"].array()
        posz[name] = events[f"{prefix}.position.z"].array()
        energy[name] = events[f"{prefix}.energy"].array()

    clusters_list = []

    for j in range(len(hits_begin_arr)):
        xs, ys, zs, ws = [], [], [], []

        for idx, coll_id in zip(
            hit_index[hits_begin_arr[j]:hits_end_arr[j]],
            collection_ID[hits_begin_arr[j]:hits_end_arr[j]]
        ):
            if coll_id not in system2name:
                continue

            sys = system2name[coll_id]
            xs.append(posx[sys][i][idx])
            ys.append(posy[sys][i][idx])
            zs.append(posz[sys][i][idx])
            ws.append(energy[sys][i][idx])

        if xs:
            clusters_list.append({
                "x": np.array(xs),
                "y": np.array(ys),
                "z": np.array(zs),
                "energy": np.array(ws)
            })

    # Flatten clusters into DataFrame
    all_x, all_y, all_z, cluster_id, all_e = [], [], [], [], []
    for cid, c in enumerate(clusters_list):
        n = len(c["x"])
        all_x.extend(c["x"])
        all_y.extend(c["y"])
        all_z.extend(c["z"])
        all_e.extend(c["energy"])
        cluster_id.extend([cid] * n)

    df = pd.DataFrame({
        "x": all_x,
        "y": all_y,
        "z": all_z,
        "cluster": cluster_id,
        "energy": all_e
    })

    return {
        "mcparticle_ids": particle_ids,
        "mcparticle_momentum": total_momentum,
        "mcparticle_energy": total_energy,
        "mcparticle_status": status_gen,
        "num_tracks": num_tracks,
        "hit_dataframe": df,
        "event_num": i
    }


for E in pion_energies:
    print(f"Processing pion energy {E} GeV")

    file = uproot.open(
        f"/users/rldohert/data/mucoll/rldohert/"
        f"pdg_211_pt_{E}_theta_15-15/"
        f"reco_pdg_211_pt_{E}_theta_15-15.root"
    )

    events = file["events"]
    n_events = len(events["MCParticles"]["MCParticles.PDG"].array())

    all_events_data = []

    for i in pion_events[E]:
        event_data = process_event(events, i, system2name)
        event_data["energy_event"] = E
        event_data["event_tag"] = "pion"
        all_events_data.append(event_data)

    # Save per-energy pickle
    with open(f"/users/rldohert/data/mucoll/rldohert/pion_energy_{E}.pkl", "wb") as f:
        pickle.dump(all_events_data, f)
    print(f"Saved pion_energy_{E}.pkl")
   