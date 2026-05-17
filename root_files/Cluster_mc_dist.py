#Playing around checking Lorentz 4 Vector


import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
import vector



#We want to use vector supposedly

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [50]
angles = [15, 85]
particles = [11, 211]
mc_median= []
mc_low = []
mc_high = []
for pid in particles:
    for a in angles:
        for e in energies:
            distances = []
            count = 0
            passes = 0
            file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
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
            print(cluster_energy)
            #for i in range((events.num_entries)):
            for i in range(5000):
                if len(cluster_energy[i]) == 1:
                    count = count+1
                    mask = (status_mc[i] == 1)
                    mx=mc_x[i][mask]
                    my=mc_y[i][mask]
                    mz=mc_z[i][mask]
                    momx=mc_momx[i][mask]
                    momy=mc_momy[i][mask]
                    momz=mc_momz[i][mask]
                    mcmass=mc_mass[i][mask]
                    cx = cluster_x[i][0] #mind you, only works for 1 cluster
                    cy = cluster_y[i][0]
                    cz = cluster_z[i][0]
                    #I am now going to vectorize and use three vectors
                    mc_arr = np.array([mx[0], my[0], mz[0]])
                    cl_arr = np.array([cx, cy, cz])
                    cos_angle = np.dot(mc_arr, cl_arr) / (np.linalg.norm(mc_arr) * np.linalg.norm(cl_arr))
                    cos_angle = np.clip(cos_angle, -1, 1)  # avoid floating point errors
                    angular_distance = np.arccos(cos_angle)
                    distances.append(angular_distance)
            print(count)
            #Now we want to print 98% for each of the things for 50 
            distance_array = np.array(distances)
            median = np.median(distance_array)
            bins = 30
            q95 = np.percentile(distance_array, [95])
            print(f"For energy {e}, particle {pid}, angle {a}, 95% is {q95}")
            plt.hist(distance_array, bins=bins, edgecolor = 'black')
            plt.xlabel(f"One Cluster Distance")
            plt.ylabel("Count")
            plt.axvline(
                median,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f"Median = {median:.2f}"
            )
            plt.legend()
            plt.title(f"One Cluster Distance from MC particle for {e}, {pid}, {a}")
            plt.tight_layout()
            plt.savefig(f"distance_from_MC_{e}_{pid}_{a}.pdf")
            plt.close()