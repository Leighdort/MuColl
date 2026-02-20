#Here I am going to look at energy differences with one cluster
#Energy difference 1 cluster
#Energy difference between leading and secondary for bib when 1 cluster
#Submit_bibnbib.sh

# Widthroot_fast.py
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
#WARNING
#SHOULD PRINT AS GOING TO REDUCE MEMORY AND TIME
system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [2, 10, 50]
choices = [2, 10, 50] #you don't need both, I just have both 
dif_median = []
dif_low = []
dif_high = []
dif_rmedian= []
dif_rlow = []
dif_rhigh = []
dlead_median = []
dlead_low = []
dlead_high = []
dleadp_median = []
dleadp_low = []
dleadp_high = []
nclus_median = []
nclus_low = []
nclus_high = []
for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    file_nobib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    events_nobib = file_nobib["events"]
    events_bib = file_bib["events"]
    pandora_clusters_nobib = events_nobib["PandoraClusters"]
    pandora_clusters_bib = events_bib["PandoraClusters"]
    nobib_energy = pandora_clusters_nobib["PandoraClusters.energy"].array()
    bib_energy = pandora_clusters_bib["PandoraClusters.energy"].array()
    eclus_dif = [] # difference btwn bib and no bib leading array
    eclus_rdif = [] #restricting the domain for what we're looking at 
    eclus_lead_dif = [] #leading vs second leading bib 
    eclus_lead_dif_p = []
    num_clus = [] #number of bib clusters
    for i in range(events_nobib.num_entries):
        eclus_nobib = 0
        eclus_leadbib = 0
        eclus_secbib = 0
        eclus_bib_total = 0
        num_bib_clus = 0 #num bib clusters 
        if i % 1000 == 0:
            print(f"Event {i}")
        if len(nobib_energy[i]) == 1:
            eclus_nobib = nobib_energy[i][0] # should test this
            these = np.asarray(bib_energy[i])
            if len(these) >= 2:
                eclus_leadbib = these.max()
                eclus_bib_total = np.sum(these)
                eclus_secbib = np.partition(these, -2)[-2] #the second largest value (-2) and extracts it 
            else:
                continue
            #eclus_leadbib = np.max(bib_energy[i])
            #eclus_secbib = sorted(set(bib_energy[i]))[-2]
            num_bib_clus = len(bib_energy[i])
            eclus_dif.append(eclus_leadbib - eclus_nobib)
            eclus_lead_dif.append(eclus_leadbib - eclus_secbib)
            proportion = ((eclus_leadbib - eclus_secbib)/ (eclus_bib_total))
            if proportion > 0.1:
                eclus_rdif.append(eclus_leadbib - eclus_nobib)
            eclus_lead_dif_p.append(proportion)
            num_clus.append(num_bib_clus)
    #Ok now after all events are accounted for
    eclus_dif = np.array(eclus_dif)
    eclus_lead_dif = np.array(eclus_lead_dif)
    eclus_lead_dif_p = np.array(eclus_lead_dif_p)
    eclus_rdif = np.array(eclus_rdif)
    num_clus = np.array(num_clus)
    q16, q84 = np.percentile(eclus_dif, [16, 84])
    median = np.median(eclus_dif)
    dif_low.append(median - q16)
    dif_high.append(q84 - median)
    dif_median.append(median)

    #I will likely have to adjust binning but this is temporary
    #bins = np.arange(np.min(eclus_dif), np.max(eclus_dif) + 2) - 0.5
    bins = 50
    plt.hist(eclus_dif, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Difference between Nonbib / bib Cluster Energy (GeV)")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Difference in Energy with Bib {num} GeV Pions")
    plt.tight_layout()
    plt.savefig(f"cluster_dif_pions_bib_log{num}GeV.pdf")
    plt.close()
    #Now we're looking 
    
    q16, q84 = np.percentile(eclus_rdif, [16, 84])
    median = np.median(eclus_rdif)
    dif_rlow.append(median - q16)
    dif_rhigh.append(q84 - median)
    dif_rmedian.append(median)

    bins = 50
    plt.hist(eclus_rdif, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Difference between Nonbib / bib Cluster Energy (GeV) ")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Difference in Energy with Bib {num} GeV Pions (lead - second > 0.1) ")
    plt.tight_layout()
    plt.savefig(f"cluster_dif_pions_bib_log_restrict{num}GeV.pdf")
    plt.close()
    # Now look at no log
    bins = 50
    plt.hist(eclus_rdif, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Difference between Nonbib / bib Cluster Energy (GeV) ")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Difference in Energy with Bib {num} GeV Pions (lead - second > 0.1) ")
    plt.tight_layout()
    plt.savefig(f"cluster_dif_pions_bib_restrict{num}GeV.pdf")
    plt.close()
    #Repeat for lead/secondary bib proportion
    q16, q84 = np.percentile(eclus_lead_dif_p, [16, 84])
    median = np.median(eclus_lead_dif_p)
    dleadp_low.append(median - q16)
    dleadp_high.append(q84 - median)
    dleadp_median.append(median)
    #I will likely have to adjust binning but this is temporary
    #bins = np.arange(np.min(eclus_lead_dif), np.max(eclus_lead_dif) + 2) - 0.5
    bins = 50
    plt.hist(eclus_lead_dif_p, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Energy Difference (GeV) between Leading and Secondary Bib Cluster / Total Energy")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Difference in Leading and Secondary Cluster with Bib {num} GeV Pions")
    plt.tight_layout()
    plt.savefig(f"bibclus_dif_pions_bib_p{num}GeV.pdf")
    plt.close()


    # Repeat for lead/secondary bib
    q16, q84 = np.percentile(eclus_lead_dif, [16, 84])
    median = np.median(eclus_lead_dif)
    dlead_low.append(median - q16)
    dlead_high.append(q84 - median)
    dlead_median.append(median)
    #I will likely have to adjust binning but this is temporary
    #bins = np.arange(np.min(eclus_lead_dif), np.max(eclus_lead_dif) + 2) - 0.5
    bins = 50
    plt.hist(eclus_lead_dif, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Energy Difference (GeV) between Leading and Secondary Bib Cluster")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Difference in Leading and Secondary Cluster with Bib {num} GeV Pions")
    plt.tight_layout()
    plt.savefig(f"bibclus_dif_pions_bib{num}GeV.pdf")
    plt.close()
    #Repeat for number of clusters
    q16, q84 = np.percentile(num_clus, [16, 84])
    median = np.median(num_clus)
    nclus_low.append(median - q16)
    nclus_high.append(q84 - median)
    nclus_median.append(median)
    #I will likely have to adjust binning but this is temporary
    bins = np.arange(np.min(num_clus), np.max(num_clus) + 2) - 0.5
    plt.hist(num_clus, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Number of Bib Clusters given 1 Normal Cluster")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Number of Bib Clusters from 1 Nonbib Cluster {num} GeV Pions")
    plt.tight_layout()
    plt.savefig(f"nbibclus_pions_bib{num}GeV.pdf")
    plt.close()

#Now we make the Summary Graph 
plt.errorbar(choices, dif_median, yerr=[dif_low, dif_high], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Difference between singular Nobib cluster with Bib")
plt.title("Difference in Energy between Leading Bib Cluster and One Cluster Event")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_clusdiff_bib.pdf")
plt.close()

#Second clustering Graph
#Now we make the Summary Graph 
plt.errorbar(choices, dif_rmedian, yerr=[dif_rlow, dif_rhigh], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Difference between singular Nobib cluster with Bib")
plt.title("Difference in Energy between Leading Bib Cluster and One Cluster Event >0.1")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_clusdiff_restrictbib.pdf")
plt.close()

#Second summary graph, difference between leading and secondary cluster w/ bib
plt.errorbar(choices, dlead_median, yerr=[dlead_low, dlead_high], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Energy Difference (GeV) between two bib Leading Clusters")
plt.title("Difference in Energy between two bib Leading Clusters given original 1 Cluster Event")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_leaddiff_bib.pdf")
plt.close()
#Third summary graph, number of bib clusters given 1 normal cluster
plt.errorbar(choices, nclus_median, yerr=[nclus_low, nclus_high], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Number of Bib Clusters")
plt.title("Number of Bib Clusters given original 1 Cluster Event")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_nbibclus_bib.pdf")
plt.close()

#Second summary graph, difference between leading and secondary cluster w/ bib
plt.errorbar(choices, dleadp_median, yerr=[dleadp_low, dleadp_high], fmt='s', alpha= 0.6, capsize=4, label="Pion particle Gun")
plt.xlabel("Beam Energy")
plt.ylabel("Median Energy Difference (GeV) between two bib Leading Clusters / total Energy")
plt.title("Difference in Energy between two bib Leading Clusters given original 1 Cluster Event")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_pleaddiff_bib.pdf")
plt.close()