


#Here i am going to try to find the width for the nobib file

#This is finding mc + locating currect clusters



#Things to bound w/
#1 cluster
#Number of events that do this
#Falls within bounds
#Number of events that do this

#Warning may fail with 0s and Division be Careful!
#This is nobib 
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [50]
choices = [50]

all_median = []
all_low = []
all_high = []

for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    widths = []
    num_events_1_clus = 0
    passes = 0
    if num == 2:
        bound = 0.05
    if num == 10:
        bound = 0.02
    if num == 50:
        bound = 0.06
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    events = file["events"]
    pandora_clusters = events["PandoraClusters"]
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
    pandora_clusters_hits = events["_PandoraClusters_hits"]
    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
    posx = {}
    posy = {}
    posz = {}
    energy_map = {}
    #I need the theta phi of mc and of clusters 
    mcparticles = events["MCParticles"]
    #we want a status of 1 
    status_mc = mcparticles["MCParticles.generatorStatus"].array()
    #do I want vertex or endpoint? I presume endpoint and we will see both
    #I will try it first with endpoint
    mc_x = mcparticles["MCParticles.endpoint.x"].array()
    mc_y = mcparticles["MCParticles.endpoint.y"].array()
    mc_z = mcparticles["MCParticles.endpoint.z"].array()
    mc_momx = mcparticles["MCParticles.momentum.x"].array()
    mc_momy = mcparticles["MCParticles.momentum.y"].array()
    mc_momz = mcparticles["MCParticles.momentum.z"].array()
    mc_mass = mcparticles["MCParticles.mass"].array()
    #Now I want cluster energy
    #Also I want cluster theta, phi
    cluster_x=pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y=pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z=pandora_clusters["PandoraClusters.position.z"].array()
    cluster_energy=pandora_clusters["PandoraClusters.energy"].array()

    #Let's right now just filter
    for name in real_systems:
        prefix = f"{name}/{name}"
        posx[name] = events[f"{prefix}.position.x"].array()
        posy[name] = events[f"{prefix}.position.y"].array()
        posz[name]   = events[f"{prefix}.position.z"].array()
        energy_map[name] = events[f"{prefix}.energy"].array()
    for i in range(events.num_entries):
        if i % 1000 == 0:
            print(f"Event {i}")
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i]) == 1:
            num_events_1_clus +=1 
            mask = (status_mc[i] == 1)
            mx=mc_x[i][mask]
            my=mc_y[i][mask]
            mz=mc_z[i][mask]
            momx=mc_momx[i][mask]
            momy=mc_momy[i][mask]
            momz=mc_momz[i][mask]
            mcmass=mc_mass[i][mask]
            mc_r = np.sqrt(mx**2 + my**2 + mz**2)
            mc_theta = np.arccos(mz / mc_r) #these may all be in radians
            mc_phi = np.arctan2(my, mx)
            cx = cluster_x[i][0] #mind you, only works for 1 cluster
            cy = cluster_y[i][0]
            cz = cluster_z[i][0]
            c_r = np.sqrt(cx**2 + cy**2 + cz**2)
            c_theta = np.arccos(cz / c_r)
            c_phi = np.arctan2(cy, cx)
            #distance = np.sqrt(mc_r**2 + c_r**2 * (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi-c_phi) + np.cos(c_theta)*np.cos(mc_theta)))
            #Now we can try to do angular distance
            cosang = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
            #Now we need to clip it
            cosang = np.clip(cosang, -1.0, 1.0)
            angular_distance = np.arccos(cosang)
            #This is important because this tells us the angle
            #Angle and making sure momentum is in the same direction is more important
            other_angle = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
            other_dist = np.sqrt(mc_r**2 + c_r**2 - 2*mc_r*c_r*other_angle)
            #We will be using angular_distance
            if angular_distance <= bound:

                #Now we are looking at width
                lo = hits_begin_all[i][0]
                hi = hits_end_all[i][0]
                hit_index = hit_index_all[i]
                idxs = hit_index[lo:hi]
                collection_ID = collectionID_all[i]
                sysIDs = collection_ID[lo:hi]
                if len(idxs) == 0:
                    continue
                sysnames = np.vectorize(system2name.get)(sysIDs)
                if 'None' in sysnames.astype(str):
                    #print(f"Skipping cluster {j} in event {i} at energy {num} GeV due to unknown system ID")
                    continue
                
                mask = (sysnames != "Skip")
                sysnames = sysnames[mask]
                idxs = idxs[mask]
                if len(sysnames) == 0:
                    continue
                # Build arrays of hit info in one pass
                xs = np.array([posx[s][i][idx] for s, idx in zip(sysnames, idxs)])
                ys = np.array([posy[s][i][idx] for s, idx in zip(sysnames, idxs)])
                zs = np.array([posz[s][i][idx] for s, idx in zip(sysnames, idxs)])
                weights = np.array([energy_map[s][i][idx] for s, idx in zip(sysnames, idxs)])
    
                if weights.sum() == 0:
                    continue
    
                # Weighted centroid (vectorized)
                x_c = np.sum(xs * weights) / np.sum(weights)
                y_c = np.sum(ys * weights) / np.sum(weights)
                z_c = np.sum(zs * weights) / np.sum(weights)
    
                # RMS in (x,y)
                r2 = (xs - x_c)**2 + (ys - y_c)**2
                r_rms = np.sqrt(np.average(r2, weights=weights))
    
                # Convert to eta-space
                mag_c = np.sqrt(x_c**2 + y_c**2 + z_c**2)
                if mag_c == 0:
                    continue
                theta_c = np.arccos(z_c / mag_c)
                eta_c = -np.log(np.tan(theta_c / 2.0))
                sigma_eta = np.arctan(r_rms / mag_c) * np.cosh(eta_c)
                widths.append(sigma_eta)
    #Now find np high med low
    if len(widths) == 0:
        print("No widths found")
        continue
    widths = np.array(widths)
    median = np.median(widths)
    q16, q84 = np.percentile(widths, [16, 84])
    all_median.append(median)
    low = median-q16
    high = q84-median
    bins = 30
    plt.hist(widths, bins=bins, edgecolor = 'black')
    plt.xlabel(r"$\sigma_\eta$")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Width of 1 Cluster with no Bib Pions {num}")
    plt.tight_layout()
    plt.savefig(f"width_nobib_{num}_pion.pdf")
    print(f"energy {num}")
    print(f"low {low}")
    print(f"high {high}")
    print(f"median {median}")

