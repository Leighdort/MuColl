#Reading the root files 
import uproot
import matplotlib.pyplot as plt
import os

print("Output_digi.root")
#file = uproot.open("/home/rldohert/tutorial2024/output_digidif.root")
file = uproot.open("/home/rldohert/tutorial2024/output_digisingle.root")
#print(file.keys())
#print(file.classnames())

#Make output directory to save the histograms
output_dir = "histogram_plotsdif"
os.makedirs(output_dir, exist_ok=True)

#This is looking at the first directory
vxd_dir = file["VXDBarrelDigitiser;1"]
print(vxd_dir.keys())
#hu, hv, hT, diffu, diffv, diffT, hitE, hitsAccepted
#These are all float histogram objects 

histE = vxd_dir["diffv;1"]
values, edges = histE.to_numpy()
bin_centers = 0.5 * (edges[:-1] + edges[1:])

plt.step(bin_centers, values, where='mid')
plt.title("VXDBarrel diffv single Histogram")
plt.xlabel("Energy")
plt.ylabel("Counts")

plot_path = os.path.join(output_dir, f"VXDBarreldiffv1.pdf")
plt.savefig(plot_path)
plt.close()
print("Histogram is hopefully there now")


#print("Output_reco.root")
#Go into the apptainer because that is where uproot is downlaoded w/out having to make a virtual enviornment 
#file = uproot.open("/home/rldohert/tutorial2024/output_reco.root")
# List all trees (usually you want the main one)
#print(file.keys())
#print(file.classnames())

#print("Output_digibefore.root")
#file = uproot.open("/home/rldohert/tutorial2024/output_digibefore.root")
#print(file.keys())
#print(file.classnames())
# To access a subobject (like a histogram) under a directory:
#vxd = file["VXDBarrelDigitiser"]  # Access the directory
#print(vxd.keys())  # List all objects in this directory

# Then access e.g. a histogram named "hu"
#hu_hist = vxd["hu"]

# To see histogram info:
#print(hu_hist)

#Ok so reco comes after digi 
#Ok lets make a newdigi and then check root
#Then check a digi after reco has been called 

#output_simtest.slcio 