
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

#This is the working one!
'''
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
pion_5_bib = [0,1,2,3,4,5,6,7,8,9]
'''
#pion_events = {
    #2: [7164, 8854, 2745, 3786, 5252, 4418, 9473, 6902, 4373, 9663, 8913, 4310, 3734, 9107, 4897, 2380, 7559, 514, 5814, 5694],
    #10: [5178, 9112, 8120, 9038, 88, 6267, 2365, 6521, 3015, 3689, 8094, 3415, 5318, 8210, 6139, 1824, 5929, 7303, 2452, 1069],
    #50: [468, 1086, 9038, 7116, 3268, 3834, 3418, 6985, 3696, 7920, 5694, 1393, 7297, 324, 9123, 271, 4441, 1280, 4224, 790],
#}
'''
pion_events = {
    2: [1568,7553,6837,9410,8006,8146,4010,9105,7811,7693,6790,7901,4635,9323,8916,7590,7684,8506,9779,5608],
    10: [7320,1630,54,1530,3183,3400,5896,4404,5841,1228,8502,3651,6668,7763,2920,4280,8706,4786,3480,9376],
    50: [7702,4875,2121,1817,9266,7122,8921,5783,9497,8726,4742,1244,4787,7103,8186,5361,6645,6766,3272,4434],
}
'''
pion_events = {
    50: [539, 1606, 2267, 2351, 2585, 2605, 3665, 3605, 3799, 3914, 4709, 4872],
    #50: [1, 3, 4, 5, 7, 8, 9, 10, 13]
}
pion_energies = [50]
def process_event(events, i, system2name):

    mcparticles = events["MCParticles"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    clusters = events["PandoraClusters"]
    pch = events["_PandoraClusters_hits"]
    pandora_clusters = events["PandoraClusters"]
    cluster_x=pandora_clusters["PandoraClusters.position.x"].array()[i]
    cluster_y=pandora_clusters["PandoraClusters.position.y"].array()[i]
    cluster_z=pandora_clusters["PandoraClusters.position.z"].array()[i]
    particle_ids = mcparticles["MCParticles.PDG"].array()[i]

    px = mcparticles["MCParticles.momentum.x"].array()[i]
    py = mcparticles["MCParticles.momentum.y"].array()[i]
    pz = mcparticles["MCParticles.momentum.z"].array()[i]
    mass = mcparticles["MCParticles.mass"].array()[i]
    mc_x = mcparticles["MCParticles.endpoint.x"].array()[i]
    mc_y = mcparticles["MCParticles.endpoint.y"].array()[i]
    mc_z = mcparticles["MCParticles.endpoint.z"].array()[i]
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

    # Map (system, hit index) -> Pandora cluster index
    hit_to_cluster = {}
    for j in range(len(hits_begin_arr)):
        sl = slice(hits_begin_arr[j], hits_end_arr[j])
        for idx, coll_id in zip(hit_index[sl], collection_ID[sl]):
            if coll_id in system2name:
                hit_to_cluster[(system2name[coll_id], int(idx))] = j
    cellid = {name: events[f"{name}/{name}.cellID"].array() for name in system2name.values()}
    parts = []
    for name in system2name.values():
        x = np.asarray(posx[name][i])
        n = len(x)
        cid = np.asarray(cellid[name][i]).astype(np.uint64) #added
        parts.append(pd.DataFrame({
            "system": name,
            "x": x,
            "y": np.asarray(posy[name][i]),
            "z": np.asarray(posz[name][i]),
            "energy": np.asarray(energy[name][i]),
            "layer": ((cid >> np.uint64(19)) & np.uint64(0x1FF)).astype(np.int64), #added
            "cluster": [hit_to_cluster.get((name, k), -1) for k in range(n)]
        }))
    all_hits = pd.concat(parts, ignore_index=True)
    all_hits["in_cluster"] = all_hits["cluster"] >= 0
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
    xs, ys, zs, ws = [], [], [], []
    for idx, coll_id in zip (hit_index, collection_ID):
        if coll_id not in system2name:
            continue
        sys = system2name[coll_id]
        xs.append(posx[sys][i][idx])
        ys.append(posy[sys][i][idx])
        zs.append(posz[sys][i][idx])
        ws.append(energy[sys][i][idx])
    whole_event = pd.DataFrame({
        "x": xs,
        "y": ys,
        "z": zs, 
        "energy": ws
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
        "energy": all_e,
    })
    return {
        "mcparticle_ids": particle_ids,
        "mcparticle_momentum": total_momentum,
        "mcparticle_energy": total_energy,
        "mcparticle_status": status_gen,
        "num_tracks": num_tracks,
        "hit_dataframe": df,
        "full_event": whole_event,
        "event_num": i,
        "Endpoint MCX": mc_x, 
        "Endpoint MCY": mc_y,
        "Endpoint MCZ": mc_z,
        "clus_x_cent": cluster_x,
        "clus_y_cent": cluster_y,
        "clus_z_cent": cluster_z,
        "all_hits": all_hits,
    }



#This is how you normally work it:
for E in pion_energies:
    print(f"Processing electron energy {E} GeV")
    #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{E}_theta_15-15_bib2/job_0/reco_output_p{E}_211_nobib0.edm4hep.root")
    #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_clustertest/reco_pdg_211_pt_50_theta_85-85_bib.root")
    file = uproot.open("/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root")
    #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_85-85_fedecheck/job_0/reco_output_p__nobib_oldalg.edm4hep.root")
    events = file["events"]
    n_events = len(events["MCParticles"]["MCParticles.PDG"].array())

    all_events_data = []

    for i in pion_events[E]:
        event_data = process_event(events, i, system2name)
        event_data["energy_event"] = E
        event_data["event_tag"] = "pion 50 nobib 85 new algorithm, high iso"
        all_events_data.append(event_data)

    # Save per-energy pickle
    with open(f"/users/rldohert/data/mucoll/rldohert/pion_10_20_85.pkl", "wb") as f:
        pickle.dump(all_events_data, f)
    print(f"Saved pion_energy_{E}no.pkl")


'''


   
file = uproot.open("/users/rldohert/data/mucoll/rldohert/pdg_211_pt_5_theta_15-15/reco_pdg_211_pt_5_theta_15-15.root")



print("Processing Bib 50 GeV")
#file = uproot.open("/users/rldohert/data/mucoll/rldohert/chunk_0/reco_output.edm4hep.root")
#file = uproot.open("/users/rldohert/data/mucoll/rldohert/chunk0_bib_211_5/reco_output.edm4hep.root")
file = uproot.open("/users/rldohert/data/mucoll/rldohert/new_sim_pt_50_0/reco_output_p50_211_bib0.edm4hep.root")
events = file["events"]
n_events = len(events["MCParticles"]["MCParticles.PDG"].array())
all_events_data = []
for i in pion_5_bib:
    event_data = process_event(events, i, system2name)
    event_data["energy_event"] = 5
    event_data["event_tag"] = "no bib 2/8/26 "
    all_events_data.append(event_data)
with open("/users/rldohert/data/mucoll/rldohert/pion_energy_50_211_bib_2_9.pkl", "wb") as f:
    pickle.dump(all_events_data, f)
    print(f"Saved pion_energy_50_nobib.pkl")

#/users/rldohert/data/mucoll/rldohert/new_sim_pt50_0/reco_output_p50_211_nobib0.edm4hep.root
'''