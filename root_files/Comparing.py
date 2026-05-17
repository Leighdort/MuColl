


#This is looking at select events:


#This is printing out specifics and also plotting resolution

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
        #bound = 0.006
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
    
    #Loop through just pions 
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
                clus_array = np.array(clus_array)
                max_energy = np.max(clus_array) #Taking the max energy
                max_energy_pos = np.argmax(clus_array)
                bib_distance.append(clus_distance[max_energy_pos])
                #print(f"Position in clus_index bib: {max_energy_pos}, cluster j index: {clus_index[max_energy_pos]}")
                fractionmc = max_energy / mc_energy[0]
                fraction_bib.append(fractionmc)
                fraction_nobib.append(clus_energy_nobib/mc_energy[0])
                nobib_distance.append(angular_distance_nobib)
    #Getting Stats
    bib_distance = np.array(bib_distance)
    nobib_distance = np.array(nobib_distance)
    fraction_bib = np.array(fraction_bib)
    fraction_nobib = np.array(fraction_nobib)
    q16, q84 = np.percentile(fraction_nobib, [16, 84])
    median = np.median(fraction_nobib)
    fnobib_low.append(median-q16)
    fnobib_high.append(q84-median)
    fnobib_median.append(median)

    q16, q84 = np.percentile(fraction_bib, [16, 84])
    median = np.median(fraction_bib)
    fbib_low.append(median-q16)
    fbib_high.append(q84-median)
    fbib_median.append(median)

    q16, q84 = np.percentile(nobib_distance, [16, 84])
    median = np.median(nobib_distance)
    dnobib_low.append(median-q16)
    dnobib_high.append(q84-median)
    dnobib_median.append(median)

    q16, q84 = np.percentile(bib_distance, [16, 84])
    median = np.median(bib_distance)
    dbib_low.append(median-q16)
    dbib_high.append(q84-median)
    dbib_median.append(median)
    
    # Common x bin choice
    xbins = 50
    ybins = 50

    # -----------------------------
    # 1. Fraction (nobib) vs nobib Distance
    xmin, xmax = 0, 0.03
    ymin, ymax = 0.95, 2

    #ybins = min(300, int(xbins * (ymax - ymin) / (xmax - xmin)))
    figsize = (6,6)
    plt.hist2d(nobib_distance, fraction_bib, bins=[xbins, ybins])
    plt.colorbar(label="Counts")
    plt.xlabel("Nobib Distance")
    plt.ylabel("Bib Cluster Energy / MC Energy")
    plt.title(f"Nobib Distance vs. Fraction Energy Bib Electrons {num}")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(f"2nobib_bibfraction_elec{num}.pdf")
    plt.close()

    figsize = (6,6)
    # -----------------------------
    # 2. Fraction (nobib) vs nobib Distance (nobib energy)
    xmin, xmax = 0, 0.03
    ymin, ymax = 0.4, 1.3

    #ybins = int(xbins * (ymax - ymin) / (xmax - xmin))

    plt.hist2d(nobib_distance, fraction_nobib, bins=[xbins, ybins])
    plt.colorbar(label="Counts")
    plt.xlabel("Nobib Distance")
    plt.ylabel("Nobib Cluster Energy / MC Energy")
    plt.title(f"Nobib Distance vs. Fraction Energy Nobib Electrons {num}")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(f"2nobib_nobibfraction_elec{num}.pdf")
    plt.close()

    figsize = (6,6)
    # -----------------------------
    # 3. Fraction (bib) vs bib Distance
    xmin, xmax = 0, 0.03
    ymin, ymax = 0.4, 1.3

    #ybins = int(xbins * (ymax - ymin) / (xmax - xmin))

    plt.hist2d(bib_distance, fraction_bib, bins=[xbins, ybins])
    plt.colorbar(label="Counts")
    plt.xlabel("Bib Distance")
    plt.ylabel("Bib Cluster Energy / MC Energy")
    plt.title(f"Bib Distance vs. Fraction Energy Bib Electrons {num}")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(f"2bib_bibfraction_elec{num}.pdf")
    plt.close()


    # -----------------------------
    # 4. Fraction bib vs fraction nobib
    xmin, xmax = 0, 2
    ymin, ymax = 0, 2

    #ybins = int(xbins * (ymax - ymin) / (xmax - xmin))  # = 45 here
    aspect = 'equal'
    plt.hist2d(fraction_bib, fraction_nobib, bins=[xbins, ybins])
    plt.colorbar(label="Counts")
    plt.xlabel("Bib Cluster Energy / MC Energy")
    plt.ylabel("Nobib Cluster Energy / MC Energy")
    plt.title(f"Bib Fraction Energy vs. Nobib Fraction Energy Electrons {num}")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(f"2bibfrac_nobibfrac_elec{num}.pdf")
    plt.close()
    '''
    #Graphs Scatter 
    plt.scatter(bib_distance, nobib_distance)
    plt.xlabel("Bib Distance")
    plt.ylabel("Nobib Distance")
    plt.title(f"Bib vs. Nobib Distance Pions {num}") 
    plt.tight_layout()
    plt.savefig(f"newbib_nobib_pion{num}.pdf")
    plt.close()

    #Now Fraction (nobib) vs nobib Distance
    plt.hist2d(nobib_distance, fraction_bib, bins=45)
    plt.colorbar(label="Counts")
    plt.xlabel("Nobib Distance")
    plt.ylabel("Bib Cluster Energy / MC Energy")
    plt.title(f"Nobib Distance vs. Fraction Energy Bib Pions {num}")
    plt.xlim(0, 0.03)
    plt.ylim(0.95, 2)
    plt.tight_layout()
    plt.savefig(f"2nobib_bibfraction_pion{num}.pdf")
    plt.close()

    #Fraction (bib) vs nobib Distance
    plt.hist2d(nobib_distance, fraction_nobib, bins=45)
    plt.colorbar(label="Counts")
    plt.xlabel("Nobib Distace")
    plt.ylabel("Nobib Cluster Energy / MC Energy")
    plt.title(f"Nobib Distance vs. Fraction Energy Nobib Pions {num}")
    plt.ylim(0.4, 1.3)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(f"2nobib_nobibfraction_pion{num}.pdf")
    plt.close()

    #Fraction (bib) vs bib Distance
    plt.hist2d(bib_distance, fraction_bib, bins=45)
    plt.colorbar(label="Counts")
    plt.xlabel("Bib Distace")
    plt.ylabel("Bib Cluster Energy / MC Energy")
    plt.title(f"Bib Distance vs. Fraction Energy Bib Pions {num}")
    plt.ylim(0.4, 1.3)
    plt.xlim(0, 0.03)
    plt.tight_layout()
    plt.savefig(f"2bib_bibfraction_pion{num}.pdf")
    plt.close()

    #Fraction bib vs fraction nobib
    plt.hist2d(fraction_bib, fraction_nobib, bins=45)
    plt.colorbar(label="Counts")
    plt.xlabel("Bib Cluster Energy / MC Energy")
    plt.ylabel("Nobib Cluster Energy / MC Energy")
    plt.title(f"Bib Fraction Energy vs. Nobib Fraction Energy Pions {num}")
    plt.ylim(0, 2)
    plt.xlim(0, 2)
    plt.tight_layout()
    plt.savefig(f"2bibfrac_nobibfrac_pion{num}.pdf")
    plt.close()
    '''
    

#Now let's make all of the summery graphs: 

#Resolution
resolution_bib = [(high + low) / 2 / med for high, low, med in zip(fbib_high, fbib_low, fbib_median)]
plt.errorbar(choices, resolution_bib, fmt='s', alpha=0.6, capsize=4, label="Electron Particle Gun Bib")
resolution_nobib = [(high + low) / 2 / med for high, low, med in zip(fnobib_high, fnobib_low, fnobib_median)]
plt.errorbar(choices, resolution_nobib, fmt='s', alpha=0.6, capsize=4, label="Electron Particle Gun Nobib")
plt.xlabel("Beam Energy")
plt.ylabel("Resolution")
plt.title("Resolution for reconstructed Energy Electron")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("2summary_res_elec.pdf")
plt.close()

#We will test and see if this works
colors = ["blue", "orange", "red"]  # 2, 10, 50 GeV
energies = [2, 10, 50]
def colored_errorbars(ax, x, y, xl, xh, yl, yh, marker, label_prefix, linestyle='-', filled=True):
    for i in range(len(x)):
        ax.errorbar(
            x[i], y[i],
            xerr=[[xl[i]], [xh[i]]],
            yerr=[[yl[i]], [yh[i]]],
            fmt=marker,
            color=colors[i],
            alpha=0.8,
            capsize=4,
            markerfacecolor=colors[i] if filled else 'none',
            markeredgecolor=colors[i],
            linestyle=linestyle,
            label = f"{label_prefix} {energies[i]} GeV"
        )

fig, ax = plt.subplots()
colored_errorbars(
    ax,
    dnobib_median, dbib_median,
    dnobib_low, dnobib_high,
    dbib_low, dbib_high,
    's',
    "Distance"
)
ax.set_xlabel("Nobib Distance")
ax.set_ylabel("Bib Distance")
ax.set_title("Bib vs Nobib Distance Electron")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("2distances_elec.pdf")
plt.close()


fig, ax = plt.subplots()
colored_errorbars(
    ax,
    dnobib_median, fnobib_median,
    dnobib_low, dnobib_high,
    fnobib_low, fnobib_high,
    's',
    "Fraction Nobib", linestyle='-', filled=True
)
colored_errorbars(
    ax,
    dnobib_median, fbib_median,
    dnobib_low, dnobib_high,
    fbib_low, fbib_high,
    'o',
    "Fraction Bib", linestyle='--', filled=False
)
ax.set_xlabel("Nobib Distance")
ax.set_ylabel("Fraction")
ax.set_title("Fraction vs Nobib Distance Electrons")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("2fractions_nobib_elec.pdf")
plt.close()


fig, ax = plt.subplots()
colored_errorbars(
    ax,
    dbib_median, fbib_median,
    dbib_low, dbib_high,
    fbib_low, fbib_high,
    's',
    "Fraction Bib"
)
ax.set_xlabel("Bib Distance")
ax.set_ylabel("Fraction Bib")
ax.set_title("Fraction Bib vs Bib Distance Electrons")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("2fraction_bibdist_elec.pdf")
plt.close()

'''
fig, ax = plt.subplots()
ax.errorbar(
    dnobib_median, dbib_median,
    xerr=[dnobib_low, dnobib_high],  # asymmetric x errors
    yerr=[dbib_low, dbib_high],       # asymmetric y errors
    fmt='s', alpha=0.6, capsize=4, label="Bib vs Nobib Distance"
)
ax.set_xlabel("Nobib Distance")
ax.set_ylabel("Bib Distance")
ax.set_title("Bib vs Nobib Distance Electron")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("distances_elec.pdf")
plt.close()

# fraction_nobib, fraction_bib vs nobib_distance
fig, ax = plt.subplots()
ax.errorbar(
    dnobib_median, fnobib_median,
    xerr=[dnobib_low, dnobib_high],
    yerr=[fnobib_low, fnobib_high],
    fmt='s', alpha=0.6, capsize=4, label="Fraction Nobib"
)
ax.errorbar(
    dnobib_median, fbib_median,
    xerr=[dnobib_low, dnobib_high],
    yerr=[fbib_low, fbib_high],
    fmt='o', alpha=0.6, capsize=4, label="Fraction Bib"
)
ax.set_xlabel("Nobib Distance")
ax.set_ylabel("Fraction")
ax.set_title("Fractions vs Nobib Distance Electron")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("fbib_nobib_electron.pdf")
plt.close()

#CHANGE THIS TO FRACTION BIB VS FRACTION BIB DISTANCE
# fraction_nobib vs bib_distance
fig, ax = plt.subplots()
ax.errorbar(
    dnobib_median, fbib_median,
    xerr=[dnobib_low, dnobib_high],
    yerr=[fbib_low, fbib_high],
    fmt='s', alpha=0.6, capsize=4, label="Fraction Bib vs nobib Distance Electron"
)
ax.set_xlabel("Nobib Distance")
ax.set_ylabel("Fraction Bib")
ax.set_title("Fraction Bib vs NoBib Distance Elextron")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("fbib_nobibdist_elec.pdf")
plt.close()
#After this select the important events and just get the numbers for those also like the cluster number
'''
#Maybe graph it too? 



#FIX LINE STYLE
#ADD KEY