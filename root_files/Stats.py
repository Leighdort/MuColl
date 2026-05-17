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
pion_events = {
    2: [1568,7553,6837,9410,8006,8146,4010,9105,7811,7693,6790,7901,4635,9323,8916,7590,7684,8506,9779,5608],
    10: [7320,1630,54,1530,3183,3400,5896,4404,5841,1228,8502,3651,6668,7763,2920,4280,8706,4786,3480,9376],
    50: [7702,4875,2121,1817,9266,7122,8921,5783,9497,8726,4742,1244,4787,7103,8186,5361,6645,6766,3272,4434],
}
mc_median= []
mc_low = []
mc_high = []
fnobib_low=[]
fnobib_high=[]
fnobib_median=[]
fbib_low=[]
fbib_high=[]
fbib_median=[]
dbib_low=[]
dbib_high=[]
dbib_median=[]
dnobib_low=[]
dnobib_high=[]
dnobib_median=[]

one_cluster = []
nobib_count = []
bib_count = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    fraction_bib = []
    fraction_nobib = []
    bib_distance = []
    nobib_distance = []
    fraction_events = []
    num_events_1_clus = 0
    passes = 0
    if num == 2:
        bound = 0.05
        #bound = .01
        #bound = 0.056
        #targets = [0.5, 1, 1.25, 2.5]
        #targets = [0.5, 1, 1.09, 1.75]
    if num == 10:
        bound = 0.02
        #targets = [0.5, 1, 1.15, 1.5]
        #targets = [0.5, 1, 1.2]
        #bound = .01
        #bound = 0.021
    if num == 50:
        bound = 0.006
        #targets = [0.8, 1.0, 1.11, 1.3]
        #bound = .01
        #targets = [0.2, 0.99, 1, 1.2]
        #bound = 0.0076
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
    one_cluster_cut = 0
    nobib_cut = 0
    bib_cut = 0 

    #Loop through just pions 
    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i]) == 1:
            one_cluster_cut += 1 
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
            mc_theta = np.arccos(mz / mc_r)[0] #these may all be in radians
            mc_phi = np.arctan2(my, mx)[0]
            #Store cluster energy if passes
            clus_array = []
            clus_index = []
            #Ok it's no longer going to work for just 1 cluster
            
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
            angular_distance_nobib = np.arccos(cosang)
            clus_energy_nobib = cluster_energy[i][0]
            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
            mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
            fractiony = clus_energy_nobib/mc_energy[0]
            clus_distance = []
            
            if angular_distance_nobib < bound:
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
                        clus_index.append(j)
                        clus_distance.append(angular_distance)
                if len(clus_array) == 0:
                    continue
                if angular_distance_nobib < bound:
                    nobib_cut += 1
                    if len(clus_array) != 0:
                        bib_cut +=1
    one_cluster.append(one_cluster_cut)
    nobib_count.append(nobib_cut)
    bib_count.append(bib_cut)

#divide nobib_count / one_cluster and plot that value
percent = np.array(nobib_count) / np.array(one_cluster)
#Let's do a pion graph now
plt.scatter(choices, one_cluster, marker='o')
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the cut (10,000 Total)")
plt.title("Pion events that make the 1 Cluster Cut")
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_oneclus_pion.pdf")
plt.close()

plt.scatter(choices, percent, marker='o')
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the cut Distance")
plt.title("Pion Fraction that make the Distance Cut")
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_bibdist_pion.pdf")
plt.close()

pion_percent = percent.copy()
pion_one_cluster = one_cluster.copy()


mc_median= []
mc_low = []
mc_high = []
fnobib_low=[]
fnobib_high=[]
fnobib_median=[]
fbib_low=[]
fbib_high=[]
fbib_median=[]
dbib_low=[]
dbib_high=[]
dbib_median=[]
dnobib_low=[]
dnobib_high=[]
dnobib_median=[]

one_cluster = []
nobib_count = []
bib_count = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    fraction_bib = []
    fraction_nobib = []
    bib_distance = []
    nobib_distance = []
    fraction_events = []
    num_events_1_clus = 0
    passes = 0
    if num == 2:
        #bound = 0.05
        #bound = .01
        bound = 0.056
        #targets = [0.5, 1, 1.25, 2.5]
        #targets = [0.5, 1, 1.09, 1.75]
    if num == 10:
        #bound = 0.02
        #targets = [0.5, 1, 1.15, 1.5]
        #targets = [0.5, 1, 1.2]
        #bound = .01
        bound = 0.021
    if num == 50:
        #bound = 0.06
        #targets = [0.8, 1.0, 1.11, 1.3]
        #bound = .01
        #targets = [0.2, 0.99, 1, 1.2]
        bound = 0.0076
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_2112_pt_{num}_theta_15-15_bib2/reco_pdg_2112_pt_{num}_theta_15-15_nobib.root")
    file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_2112_pt_{num}_theta_15-15_bib2/reco_pdg_2112_pt_{num}_theta_15-15_bib.root")
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
    one_cluster_cut = 0
    nobib_cut = 0
    bib_cut = 0 

    #Loop through just pions 
    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i]) == 1:
            one_cluster_cut += 1 
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
            mc_theta = np.arccos(mz / mc_r)[0] #these may all be in radians
            mc_phi = np.arctan2(my, mx)[0]
            #Store cluster energy if passes
            clus_array = []
            clus_index = []
            #Ok it's no longer going to work for just 1 cluster
            
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
            angular_distance_nobib = np.arccos(cosang)
            clus_energy_nobib = cluster_energy[i][0]
            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
            mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
            fractiony = clus_energy_nobib/mc_energy[0]
            clus_distance = []
            
            if angular_distance_nobib < bound:
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
                        clus_index.append(j)
                        clus_distance.append(angular_distance)
                if len(clus_array) == 0:
                    continue
                if angular_distance_nobib < bound:
                    nobib_cut += 1
                    if len(clus_array) != 0:
                        bib_cut +=1
    one_cluster.append(one_cluster_cut)
    nobib_count.append(nobib_cut)
    bib_count.append(bib_cut)

#divide nobib_count / one_cluster and plot that value
percent = np.array(nobib_count) / np.array(one_cluster)
#Let's do a pion graph now
plt.scatter(choices, one_cluster, marker='o')
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the cut (10,000 Total)")
plt.title("Neutron events that make the 1 Cluster Cut")
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_oneclus_neutron.pdf")
plt.close()

plt.scatter(choices, percent, marker='o')
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the cut Distance")
plt.title("Neutron Fraction that make the Distance Cut")
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_bibdist_neutron.pdf")
plt.close()

neutron_percent = percent.copy()
neutron_one_cluster = one_cluster.copy()

mc_median= []
mc_low = []
mc_high = []
fnobib_low=[]
fnobib_high=[]
fnobib_median=[]
fbib_low=[]
fbib_high=[]
fbib_median=[]
dbib_low=[]
dbib_high=[]
dbib_median=[]
dnobib_low=[]
dnobib_high=[]
dnobib_median=[]

one_cluster = []
nobib_count = []
bib_count = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    fraction_bib = []
    fraction_nobib = []
    bib_distance = []
    nobib_distance = []
    fraction_events = []
    num_events_1_clus = 0
    passes = 0
    if num == 2:
        #bound = 0.05
        bound = .01
        #bound = 0.056
        #targets = [0.5, 1, 1.25, 2.5]
        #targets = [0.5, 1, 1.09, 1.75]
    if num == 10:
        #bound = 0.02
        #targets = [0.5, 1, 1.15, 1.5]
        #targets = [0.5, 1, 1.2]
        bound = .01
        #bound = 0.021
    if num == 50:
        #bound = 0.06
        #targets = [0.8, 1.0, 1.11, 1.3]
        bound = .01
        #targets = [0.2, 0.99, 1, 1.2]
        #bound = 0.0076
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15_bib2/reco_pdg_11_pt_{num}_theta_15-15_nobib.root")
    file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15_bib2/reco_pdg_11_pt_{num}_theta_15-15_bib.root")
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
    one_cluster_cut = 0
    nobib_cut = 0
    bib_cut = 0 

    #Loop through just pions 
    #Let's right now just filter
    for i in range(events.num_entries):
        #Let's just right now filter for events with only 1 cluster
        if len(cluster_energy[i]) == 1:
            one_cluster_cut += 1 
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
            mc_theta = np.arccos(mz / mc_r)[0] #these may all be in radians
            mc_phi = np.arctan2(my, mx)[0]
            #Store cluster energy if passes
            clus_array = []
            clus_index = []
            #Ok it's no longer going to work for just 1 cluster
            
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
            angular_distance_nobib = np.arccos(cosang)
            clus_energy_nobib = cluster_energy[i][0]
            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
            mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
            fractiony = clus_energy_nobib/mc_energy[0]
            clus_distance = []
            
            if angular_distance_nobib < bound:
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
                        clus_index.append(j)
                        clus_distance.append(angular_distance)
                if len(clus_array) == 0:
                    continue
                if angular_distance_nobib < bound:
                    nobib_cut += 1
                    if len(clus_array) != 0:
                        bib_cut +=1
    one_cluster.append(one_cluster_cut)
    nobib_count.append(nobib_cut)
    bib_count.append(bib_cut)

#divide nobib_count / one_cluster and plot that value
percent = np.array(nobib_count) / np.array(one_cluster)
#Let's do a pion graph now
plt.scatter(choices, one_cluster, marker='o')
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the cut (10,000 Total)")
plt.title("Electron events that make the 1 Cluster Cut")
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_oneclus_electron.pdf")
plt.close()

plt.scatter(choices, percent, marker='o')
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the cut Distance")
plt.title("Electron Fraction that make the Distance Cut")
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_bibdist_electron.pdf")
plt.close()

electron_percent = percent.copy()
electron_one_cluster = one_cluster.copy()

#pion_percent, neutron_percent, electron_percent
#pion_one_cluster, neutron_one_cluster, electron_one_cluster

plt.scatter(choices, pion_percent, marker='o', label="Pion")
plt.scatter(choices, neutron_percent, marker='s', label="Neutron")
plt.scatter(choices, electron_percent, marker='^', label="Electron")
plt.xlabel("Beam Energy")
plt.ylabel("Fraction that make the Distance Cut")
plt.title("Fraction that make the Distance Cut")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_bibdist_all.pdf")
plt.close()

# Graph 2 - Events that make the 1 Cluster Cut
plt.scatter(choices, pion_one_cluster, marker='o', label="Pion")
plt.scatter(choices, neutron_one_cluster, marker='s', label="Neutron")
plt.scatter(choices, electron_one_cluster, marker='^', label="Electron")
plt.xlabel("Beam Energy")
plt.ylabel("Events that make the 1 Cluster Cut")
plt.title("Events that make the 1 Cluster Cut")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("newsummary_oneclus_all.pdf")
plt.close()