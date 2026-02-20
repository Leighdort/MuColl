#Data for Pheonix
import uproot
import json
#Need event, mcparticle, particle0, particle1 (these are objectes)
#array of Hit objects, format is [hit, hit, hit]
#Object has type "Point, Line, or Box"
#Line should have 6 coordinates [x0, y0, z0, x1, y1, z1]
#color, optional
#Geometry data: .obj, .gltf, .root, .json.gz, .json
#This data may also have to be in a json
choices = [200]
list_to_look = [1434]
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    event_indices = list_to_look
    event_idx = 1434
    mc_particles = events["MCParticles"]
    statuses = mc_particles["MCParticles.generatorStatus"].array()[event_idx]
    pdg_momentum_x= mc_particles["MCParticles.momentum.x"].array()[event_idx]
    pdg_momentum_y= mc_particles["MCParticles.momentum.y"].array()[event_idx]
    pdg_momentum_z= mc_particles["MCParticles.momentum.z"].array()[event_idx]
    masses= mc_particles["MCParticles.mass"].array()[event_idx]
    mc_pids= mc_particles["MCParticles.PDG"].array()[event_idx]
    vertex_x= mc_particles["MCParticles.vertex.x"].array()[event_idx]
    vertex_y= mc_particles["MCParticles.vertex.y"].array()[event_idx]
    vertex_z= mc_particles["MCParticles.vertex.z"].array()[event_idx]
    endpoint_x= mc_particles["MCParticles.endpoint.x"].array()[event_idx]
    endpoint_y= mc_particles["MCParticles.endpoint.y"].array()[event_idx]
    endpoint_z= mc_particles["MCParticles.endpoint.z"].array()[event_idx]
    subsystem_hitmap = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        subsystem_hitmap[name] = {
            "posx": events[f"{prefix}.position.x"].array()[event_indices],
            "posy": events[f"{prefix}.position.y"].array()[event_indices],
            "posz": events[f"{prefix}.position.z"].array()[event_indices],
            "energy_map": events[f"{prefix}.energy"].array()[event_indices],
            "times": events[f"{prefix}.time"].array()[event_indices],
        }
    clusters = events["PandoraClusters"] #this is what i added
    pandora_clusters = events["_PandoraClusters_hits"]
    hit_index_all = pandora_clusters["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters["_PandoraClusters_hits.collectionID"].array()
    # event clusterf hit ranges
    hits_begin_all = clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = clusters["PandoraClusters.hits_end"].array()
    pandora_cluster_data = {}
    for evt in event_indices:
        pandora_cluster_data[evt] = []
        begins = hits_begin_all[evt]
        ends   = hits_end_all[evt]
        for b, e in zip(begins, ends):
            b = int(b)
            e = int(e)
            pandora_cluster_data[evt].append({
                "hit_index": hit_index_all[b:e],
                "collectionID": collectionID_all[b:e]
            })
    
    #Now I want to get calo hits
    hits = []
    for i in range(len(mc_pids)):
        if statuses[i] != 0:
            continue
        hits.append({
            "type": "Line",
            "pos": [float(vertex_x[i]), float(vertex_y[i]), float(vertex_z[i]),float(endpoint_x[i]), float(endpoint_y[i]), float(endpoint_z[i])],
            "color": "#0000FF",
            "pid": float(mc_pids[i])
        })
    event_json = {
        "event number": event_idx,
        "run number": 0,
        "Hits": {
            "MC_Truth": hits
        }
    }
    print(len(hits))
    print(hits[0])
    with open(f"phoenix_E{num}_evt{event_idx}.json", "w") as f:
        json.dump([event_json], f, indent=2)
    print("finished")

#Electron energy 2, Event 2663