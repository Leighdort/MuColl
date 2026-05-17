#This is finding mc + locating currect clusters



#Things to bound w/
#1 cluster
#Number of events that do this
#Falls within bounds
#Number of events that do this

#Warning may fail with 0s and Division be Careful!
#This is bib 

#Submit_mcbibfraction.py
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
energies = [2, 10, 50]
choices = [2, 10, 50]

mc_median= []
mc_low = []
mc_high = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    fraction = []
    num_events_1_clus = 0
    passes = 0
    if num == 2:
        bound = 0.05
        #bound = .01
    if num == 10:
        bound = 0.02
        #bound = .01
    if num == 50:
        bound = 0.006
        #bound = .01
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    bib_events = file_bib["events"]
    events = file["events"]
    pandora_clusters_bib = bib_events["PandoraClusters"]
    pandora_clusters = events["PandoraClusters"]
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
    cluster_x_bib=pandora_clusters_bib["PandoraClusters.position.x"].array()
    cluster_y_bib=pandora_clusters_bib["PandoraClusters.position.y"].array()
    cluster_z_bib=pandora_clusters_bib["PandoraClusters.position.z"].array()
    cluster_energy_bib=pandora_clusters_bib["PandoraClusters.energy"].array()
    
    angular_dist = []
    regular_dist = []
    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i]) == 1:
            #num_events_1_clus +=1 
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
            #Store cluster energy if passes
            clus_array = []
            #Ok it's no longer going to work for just 1 cluster
            for j in range(len(cluster_energy_bib[i])):
                cx = cluster_x_bib[i][j] #mind you, only works for 1 cluster
                cy = cluster_y_bib[i][j]
                cz = cluster_z_bib[i][j]
                cenergy = cluster_energy_bib[i][j]
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
                #other_angle = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                #other_dist = np.sqrt(mc_r**2 + c_r**2 - 2*mc_r*c_r*other_angle)
                #We will be using angular_distance
                if angular_distance <= bound:
                    clus_array.append(cenergy)
            if len(clus_array) == 0:
                continue
            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
            passes += 1
            mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
            clus_array = np.array(clus_array)
            max_energy = np.max(clus_array)
            fractionmc = max_energy / mc_energy[0]
            fraction.append(fractionmc)
    print(f"Number of Events that fall within margin: {passes}")
    fraction = np.array(fraction)
    q16, q84 = np.percentile(fraction, [16, 84])
    median = np.median(fraction)
    mc_low.append(median-q16)
    mc_high.append(q84-median)
    mc_median.append(median)
    bins=30
    plt.hist(fraction, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Response with Bib")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Response w/ 1 Cluster within Bounds Pion Bib {num}")
    plt.tight_layout()
    plt.savefig(f"response_bib_pion{num}GeV.pdf")
    plt.close()

#Third summary graph, number of bib clusters given 1 normal cluster
plt.errorbar(choices, mc_median, yerr=[mc_low, mc_high], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Response")
plt.title("Median Response (Pion) Bib")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("newsummary_mc_bib_pion.pdf")
plt.close()