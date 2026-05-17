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
#from scipy.optimize import curve_fit

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [2, 5, 10, 15, 30, 50]
choices = [2, 5, 10, 15, 30, 50]
particle = [11]

bounds = {
    #11: {   # electron 85 -85 
    #    2: 1.40, 
    #    10: 0.09,
    #    30: 0.06,
    #    50: 0.06,
    #},
    11: { #electron 143-143
        2: 0.59,
        5: 0.05,
        10: 0.04,
        15: 0.04,
        30: 0.04,
        50: 0.04
    },
    211: {  # pion 143-143
        2: 0.41,
        10: 0.03,
        30: 0.01,
        50: 0.008
    },
    #211: { #pion 85 -85 
    #    2: 0.82,
    #    10: 0.06,
    #    30: 0.02,
    #    50: 0.02
    #}
}
for pid in particle:
    mc_median= []
    mc_low = []
    mc_high = []
    for num in energies:
        print(f"\n=== Energy {num} GeV ===")
        fraction = []
        num_events_1_clus = 0
        passes = 0
        bound = bounds[pid][num]
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{num}_theta_143-143_bib2/reco_pdg_{pid}_pt_{num}_theta_143-143_nobib.root")
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
        angular_dist = []
        regular_dist = []
        #Let's right now just filter
        #for i in range(events.num_entries):
        #for i in range((5001)):
        for i in range((events.num_entries)):
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
                    passes += 1
                    mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                    mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                    fraction_mc = cluster_energy[i][0] / mc_energy[0]
                    fraction.append(fraction_mc)
        fraction = np.array(fraction)
        q16, q84 = np.percentile(fraction, [16, 84])
        median = np.median(fraction)
        mc_low.append(median-q16)
        mc_high.append(q84-median)
        mc_median.append(median)
        #bins=30
        bins = np.linspace(0, 2, 61)  # 60 bins → need 61 edges
        plt.hist(fraction, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Cluster/MC Energy Fraction")
        plt.ylabel("Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.2f}"
        )
        plt.legend()
        plt.title(f" 143 deg given 1 Cluster PID: {pid} Nobib Energy {num}")
        plt.tight_layout()
        plt.savefig(f"5mc_143_fraction_nobib_{pid}{num}GeV.pdf")
        plt.close()

    #Third summary graph, number of bib clusters given 1 normal cluster
    plt.errorbar(choices, mc_median, yerr=[mc_low, mc_high], fmt='s', alpha= 0.6, capsize=4, label=f"{pid} particle gun")
    plt.xlabel("Beam Energy")
    plt.ylabel("Median Cluster/MC Energy")
    plt.title(f"143 deg Median Cluster/MC Energy 1 Nobib Cluster Pid: {pid} ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"5mc_143{pid}.pdf")
    plt.close()


'''
    #Now trying to log fit the data?
    #mc median against choices!
    x_data = np.array(choices)
    y_data = np.array(mc_median)
    def log_func(x, a, c): return a * np.log(x) + c

    # 2. Fit
    popt, _ = curve_fit(log_func, x_data, y_data)
    a_opt, c_opt = popt # Optimized parameters

    # 3. Plot
    plt.scatter(x_data, y_data)
    plt.plot(x_data, log_func(x_data, *popt), 'r')
    plt.tight_layout()
    plt.savefig(f"3test.pdf")
    plt.close()
    #Mc_fraction_nobib.py

'''
