#Printing files
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

#Printing time w/ fixed margins and median 
#Reminder that this is median not mean 

electron_ecal_mean = [0.3783104419708252, 0.500889778137207, 1.0601143836975098, 1.20090913772583, 4.6545329093933105, 6.734328746795654, 7.625155448913574, 8.08724594116211]
electron_ecal_top = [0.6691693401336669, 0.6477542114257813, 0.40940813064575154, 2.7356907081604, 3.4400527763366693, 2.2734413337707515, 1.6855512619018551, 1.373823509216308]
electron_ecal_bottom = [0.12215775489807129, 0.17532859802246092, 0.5781846809387208, 0.3830242347717284, 3.2200663566589354, 3.1586950492858885, 2.5909651947021484, 2.1751483917236323]
electron_hcal_mean = [9.073324203491211, 9.093955993652344, 9.094072341918945, 9.074875831604004, 8.987945556640625, 8.96776294708252, 8.962562561035156, 8.959850311279297]
electron_hcal_top = [0.1345323944091792, 0.16369377136230412, 0.18562488555908274, 0.19740604400634787, 0.0971881866455071, 0.024688453674317046, 0.0119269943237299, 0.00828327178955135]
electron_hcal_bottom = [0.09816753387451094, 0.10716300964355518, 0.11230480194091719, 0.09481544494628835, 0.02507347106933544, 0.010748748779297657, 0.008050575256348225, 0.006775207519531534]

#Now pions
pion_ecal_mean = [4.9331889152526855, 5.693554401397705, 6.841360569000244, 7.5291852951049805, 8.95492172241211, 9.329084396362305, 9.483776092529297, 9.587979316711426]
pion_ecal_top = [3.3381611061096166, 3.010658702850341, 2.2964994239807126, 1.909184837341309, 0.849562911987304, 0.5431314468383786, 0.4195354843139647, 0.3325027465820316]
pion_ecal_bottom = [4.770887699127197, 5.5827008247375485, 6.667044811248779, 7.285134124755859, 6.296203804016111, 4.705866813659668, 4.015409278869629, 2.985259017944336]
pion_hcal_mean = [9.018901824951172, 8.978691101074219, 8.972211837768555, 8.968003273010254, 8.961185455322266, 8.959522247314453, 8.95848274230957, 8.957881927490234]
pion_hcal_top = [0.8258917617797845, 0.19790115356445348, 0.022877311706542613, 0.005801200866699219, 0.011472702026367188, 0.013163566589355469, 0.0142059326171875, 0.014798316955566548]
pion_hcal_bottom = [0.04216072082519595, 0.009429626464843466, 0.01013847351074304, 0.008403778076171875, 0.0062961578369140625, 0.005977935791015909, 0.005605201721191833, 0.005480575561524148]

energies = [1,2,5,10,50,100,150,200]
plt.errorbar(energies, pion_ecal_mean, yerr=[pion_ecal_bottom, pion_ecal_top], fmt='s', alpha=0.6, capsize=4, label="Pions")
plt.errorbar(energies, electron_ecal_mean, yerr=[electron_ecal_bottom, electron_ecal_top], fmt='s', alpha=0.6, capsize=4, label="Electrons")
plt.xlabel("Beam Energy")
plt.ylabel("Median Last time in the Ecal (ns)")
plt.title("Median Last time in the Ecal vs. Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_last_ecal.pdf")
plt.close()

plt.errorbar(energies, pion_hcal_mean, yerr=[pion_hcal_bottom, pion_hcal_top], fmt='s', alpha=0.6, capsize=4, label="Pions")
plt.errorbar(energies, electron_hcal_mean, yerr=[electron_hcal_bottom, electron_hcal_top], fmt='s', alpha=0.6, capsize=4, label="Electrons")
plt.xlabel("Beam Energy")
plt.ylabel("Median First time in the Hcal (ns)")
plt.title("Median First time in the Hcal vs. Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_first_hcal.pdf")
plt.close()

'''
# Electrons
average_energy = [0, 0, 0, 0, 0, 0, 0.0005786353, 0.00069044356]
energy_low     = [0, 0, 0, 0, 0, 0, 0.00020144523005001248, 0.0]
energy_high    = [0, 0, 0, 0, 0, 0, 0.00020144520094618197, 0.0]
average_ratio  = [0, 0, 0, 0, 0, 0, 1.9145516e-06, 1.0076259e-06]
ratio_low      = [0, 0, 0, 0, 0, 0, 9.567573238200566e-07, 0.0]
ratio_high     = [0, 0, 0, 0, 0, 0, 9.56757494350313e-07, 0.0]
# Pions
average_energy_pion = [0, 0.0009443901, 0.0011900663, 0.0006867727, 0.0009141404, 0.0012022969, 0.0012625587, 0.0013574514]
energy_low_pion     = [0, 0.0, 0.0006341926916502416, 0.00043128975317813456, 0.0005890181101858616, 0.000849611267913133, 0.0008869608282111585, 0.0009783530433196575]
energy_high_pion    = [0, 0.0, 0.0005645530484616758, 0.0015749703138135374, 0.0030316052865236986, 0.0033819196559488783, 0.0038913877075538034, 0.0040871965372934835]
average_ratio_pion  = [0, 0.00013167952, 5.3859763e-05, 1.7581797e-05, 4.5665765e-06, 2.8865634e-06, 2.1298388e-06, 1.7186462e-06]
ratio_low_pion      = [0, 0.0, 3.153374957037158e-05, 1.1355171391187469e-05, 2.989639065162919e-06, 2.039994606093387e-06, 1.5118764167709741e-06, 1.240226233676367e-06]
ratio_high_pion     = [0, 0.0, 7.036432260065339e-05, 7.709281002462375e-05, 1.607823593076318e-05, 8.984782471088693e-06, 6.722134439769422e-06, 5.332209948392105e-06]
energies = [1,2,5,10,50,100,150,200]

plt.errorbar(energies, average_energy_pion, yerr=[energy_low_pion, energy_high_pion], fmt='s', alpha= 0.6, capsize=4, label="Pions")
plt.errorbar(energies, average_energy, yerr=[energy_low, energy_high], fmt='o', alpha=0.6, capsize=4, label="Electrons")
plt.xlabel("Beam Energy")
plt.ylabel("Median Punch Through Energy")
plt.title("Median Punch Through Energy versus Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_missed_energy.pdf")
plt.close()

plt.errorbar(energies, average_ratio_pion, yerr=[ratio_low_pion, ratio_high_pion], fmt='s', alpha= 0.6, capsize=4, label="Pions")
plt.errorbar(energies, average_ratio, yerr=[ratio_low, ratio_high], fmt='o', alpha=0.6, capsize=4, label="Electrons")
plt.xlabel("Beam Energy")
plt.ylabel("Median Punch Through Energy / Total")
plt.title("Median Punch Through Energy / Total versus Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_missed_ratio_energy.pdf")
plt.close()
'''


'''
#Printing width (new width after fixed 50 )
energies = [1,2,5,10,50,100,150,200]
electron_median = [0.03034,0.02984,0.02911,0.02920,0.03037,0.03070,.03087,.03103]
electron_upper = [0.01046,0.00609,0.00379,0.00273,0.00192,0.00153,.00150,.00148]
electron_lower = [0.00600,0.00430,0.00336,0.00333,0.00123,0.00096,.00083,.00079]
pion_median = [0.10532,0.11018,0.12489,0.13053,0.14186,0.14620,.14767,.14860]
pion_upper =[0.05137,0.04627,0.04357,0.04102,0.03307,0.03146,.03009,.03003]
pion_lower = [0.03833,0.03783,0.04373,0.04303,0.04163,0.04025,.03972,.04211]

plt.errorbar(energies, pion_median, yerr=[pion_lower, pion_upper], fmt='s', alpha= 0.6, capsize=4, label="Pions")
plt.errorbar(energies, electron_median, yerr=[electron_lower, electron_upper], fmt='o', alpha=0.6, capsize=4, label="Electrons")
plt.xlabel("Beam Energy")
plt.ylabel("Median Cluster Width")
plt.title("Median Cluster Width versus Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_width.pdf")
plt.close()
''''''
energies = [1, 2, 5, 10, 50, 100, 150, 200]
electron_info_miss = [0,0,0,0,0,0,0.0005786353,0.00069044356]
electron_std_miss = [0, 0, 0, 0, 0, 0,0.00029624294,0]
electron_info_ratio = [0,0,0,0,0,0,1.91455163e-06, 1.0076259e-06]
electron_std_ratio = [0,0,0,0,0,0,1.4069961e-06,0]
pion_info_miss = [0, 0.0009443901, 0.001410144, 0.0013133102, 0.0024419392, 0.0029912659, 0.0033331513, 0.0035017321]
pion_std_miss = [0, 0.0, 0.0012682051, 0.0015092281, 0.0060342625, 0.006743244, 0.0065694037, 0.006882778]
pion_info_ratio = [0, 0.00013167952, 0.0000674254, 0.00007686458, 0.000018252635, 0.00001738953, 0.0000070982196, 0.00000632725]
pion_std_ratio = [0, 0.0, 0.000055184868, 0.0002103445, 0.00007696671, 0.00023475387, 0.000027816512, 0.00004324816]
types = 2032
electrons = [0,0,0,0,0,0,9,7]
pions = [0,3,82,393,6716,18988,33050,45838]

#We should also add particle type
plt.errorbar(energies, electron_info_miss, yerr=electron_std_miss, fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_info_miss, yerr=pion_std_miss, fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy (GeV)")
plt.ylabel("Missed Energy")
plt.title("Missed Energy vs Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("missed_energy_vs_energy.pdf")
plt.close()

plt.errorbar(energies, electron_info_ratio, yerr=electron_std_ratio, fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_info_ratio, yerr=pion_std_ratio, fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy (GeV)")
plt.ylabel("Missed / Total Ratio")
plt.title("Missed/Total Energy Ratio vs Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("missed_ratio_vs_energy.pdf")
plt.close()

plt.plot(energies, electrons, 'o-', label="Electrons")
plt.plot(energies, pions, 's-', label="Pions")
plt.xlabel("Beam Energy (GeV)")
plt.ylabel(f"Particle Counts (Type {types})")
plt.title("Particle Counts vs Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("particle_counts_vs_energy.pdf")
plt.close()
'''
'''
energies = [1, 2, 5, 10, 50, 100, 150, 200]
electron_info = [0.03281,0.03096,0.02966,0.02964,0.03233,0.03228,0.03230,0.03247]
electron_std = [0.01034,0.00582,0.00571,0.00635,0.00843,0.00796,0.00703,0.00703]
pion_info = [0.11279,0.11553,0.12668,0.13100,0.13880,0.14297,0.14452,0.14476]
pion_std = [0.04863,0.04516,0.04445,0.04274,0.03766,0.03526,0.03408,0.03431]

plt.errorbar(energies, electron_info, yerr=electron_std, fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_info, yerr=pion_std, fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy")
plt.ylabel("Width")
plt.title("Average width per energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_width.pdf")
plt.close()
'''
'''
energies = [1, 2, 5, 10, 50, 100, 150, 200]
# Mean ECal energies
elec_mean_ecal = [0.6207181811332703, 0.8626291751861572, 1.4255902767181396, 2.1203510761260986,
             4.791934967041016, 6.3585333824157715, 7.190113067626953, 7.708284378051758]
# Mean HCal energies
elec_mean_hcal = [9.811707496643066, 9.339003562927246, 9.558516502380371, 9.4028959274292,
             9.043529510498047, 8.988040924072266, 8.966110229492188, 8.961085319519043]
# Stddev ECal energies
elec_std_ecal = [0.8839275240898132, 1.144112467765808, 1.6336638927459717, 2.149731159210205,
            2.7776906490325928, 2.4482004642486572, 2.093783140182495, 1.7687077522277832]
# Stddev HCal energies
elec_std_hcal = [6.8369059562683105, 1.7655836343765259, 4.306597709655762, 3.0265491008758545,
            0.5549065470695496, 0.875796377658844, 0.017643479630351067, 0.010108085349202156]

#Ok now the same for pions;

# Mean ECal energies
pion_mean_ecal = [4.607046127319336, 5.0492024421691895, 5.725510120391846, 6.150203704833984,
             7.2506513595581055, 7.739747047424316, 7.96605920791626, 8.214613914489746]
# Mean HCal energies
pion_mean_hcal = [10.17436695098877, 9.502426147460938, 9.039555549621582, 8.983073234558105,
             8.959487915039062, 8.959606170654297, 8.958731651306152, 8.958168029785156]
# Stddev ECal energies
pion_std_ecal = [3.211388111114502, 3.3588998317718506, 3.425992727279663, 3.496840238571167,
            3.426414966583252, 3.2594592571258545, 3.15775990486145, 2.9915571212768555]
# Stddev HCal energies
pion_std_hcal = [4.932833194732666, 3.1433820724487305, 1.1571003198623657, 0.43229490518569946,
            0.022027665749192238, 0.013457683846354485, 0.013496596366167068, 0.01357179507613182]

plt.errorbar(energies, elec_mean_ecal, yerr=elec_std_ecal, fmt='o', capsize=4, label="Electrons Average Ecal End")
plt.errorbar(energies, elec_mean_hcal, yerr=elec_std_hcal, fmt='s', capsize=4, label="Electrons Average Hcal Start")
plt.errorbar(energies, pion_mean_ecal, yerr=pion_std_ecal, fmt='o', capsize=4, label="Pions Average Ecal End")
plt.errorbar(energies, pion_mean_hcal, yerr=pion_std_hcal, fmt='s', capsize=4, label="Pions Average Hcal Start")
plt.xlabel("Beam Energy")
plt.ylabel("Time")
plt.title("Average Time for Exit for Entry for a Cluster")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_time_cluster.pdf")
plt.close()
'''