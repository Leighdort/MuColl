#Distance to truth

#This is assuming cluster position is the energy weighted center
#---------> I believe this is correct

#Submitmc_dist.sh
#What’s the average non-bib distance from the cluster to mc particle (looking at weighted cluster center) ** angular distance **
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
print("Here")
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
#energies = [2, 10, 30, 50]
#choices = [2, 10, 30, 50]
energies = [2, 5, 10, 15, 30, 50]
choices = [2, 5, 10, 15, 30, 50]
angular_med = []
angular_low = []
angular_high = []
r_med = []
r_low = []
r_high = []
one_clusters = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_143-143_bib2/reco_pdg_11_pt_{num}_theta_143-143_nobib.root")
    #file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    events = file["events"]
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
    #mc_x = mcparticles["MCParticles.momentumAtEndpoint.x"].array()
    #mc_y = mcparticles["MCParticles.momentumAtEndpoint.y"].array()
    #mc_z = mcparticles["MCParticles.momentumAtEndpoint.z"].array()
    #Now I want cluster energy
    #Also I want cluster theta, phi
    cluster_x=pandora_clusters["PandoraClusters.position.x"].array()
    cluster_y=pandora_clusters["PandoraClusters.position.y"].array()
    cluster_z=pandora_clusters["PandoraClusters.position.z"].array()
    cluster_energy=pandora_clusters["PandoraClusters.energy"].array()
    angular_dist = []
    regular_dist = []
    count = 0
    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i]) == 1:
            count += 1
            mask = (status_mc[i] == 1)
            mx=mc_x[i][mask]
            my=mc_y[i][mask]
            mz=mc_z[i][mask]
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
            angular_dist.append(angular_distance)
            regular_dist.append(other_dist)
    one_clusters.append(count)
    #Now I am finding the mean for each energy
    angular_dist = np.array(angular_dist)
    regular_dist = np.array(regular_dist)
    q16, q84 = np.percentile(angular_dist, [16, 84])
    median = np.median(angular_dist)
    angular_low.append(median-q16)
    angular_high.append(q84 - median)
    angular_med.append(median)
    #bins = 30
    bins = np.linspace(0, 0.1, 31)
    plt.hist(angular_dist, bins=bins, edgecolor = 'black')
    plt.xlim(0, 0.1) 
    plt.xlabel(f"Angular Distance from Center of Cluster Momentum")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Angular Distance (143) from ClusCenter (1 Clus) to MomentumEnd {num} GeV Electrons")
    plt.tight_layout()

    plt.savefig(f"6ang__143_dist_momend_elec{num}GeV.pdf")
    plt.close()
    p98 = np.percentile(angular_dist, 98)
    print(f"For {num} GeV, 98th percentile is {p98}")
    print("Electron 143 degrees")

    '''
    q16, q84 = np.percentile(regular_dist, [16, 84])
    median = np.median(regular_dist)
    r_low.append(median-q16)
    r_high.append(q84 - median)
    r_med.append(median)
    bins = 30
    plt.hist(regular_dist, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Spherical Distance from Center of Cluster to Momentum Neutrons")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Spherical Distance from Center of Cluster to MomentumEnd {num} GeV Neutron")
    plt.tight_layout()
    plt.savefig(f"sph_dist_momend_neut{num}GeV.pdf")
    plt.close()
    '''
plt.errorbar(choices, angular_med, yerr=[angular_low, angular_high], fmt='s', alpha= 0.6, capsize=4, label="Electron particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Angular Distance")
plt.title("Angular Distance 143 from 1 Cluster to MC Particle End Electrons")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("5summary_143_angmomend_bib_electron.pdf")
plt.close()

plt.scatter(choices, one_clusters)
plt.xlabel("Beam Energy")
plt.ylabel("Number of Clusters")
plt.title("Clusters that make the 1 Cluster Cut")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("one_pion_cluster.pdf")
plt.close()

'''
plt.errorbar(choices, r_med, yerr=[r_low, r_high], fmt='s', alpha= 0.6, capsize=4, label="Neutron particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Spherical Distance")
plt.title("Spherical Distance from 1 Cluster to MC Particle MomentumEnd Neutrons")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_sphmomend_bib_neut.pdf")
plt.close()
'''