#This is testing that my root files are correct
import uproot
print("reco_pdg_11_pt_100_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_100_theta_15-15/reco_pdg_11_pt_100_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_11_pt_10_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_10_theta_15-15/reco_pdg_11_pt_10_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_11_pt_150_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_150_theta_15-15/reco_pdg_11_pt_150_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_11_pt_1_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_1_theta_15-15/reco_pdg_11_pt_1_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_11_pt_200_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_200_theta_15-15/reco_pdg_11_pt_200_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries) 
#The above are successful 

print("reco_pdg_11_pt_2_theta_15-15.root") #It failed on this one
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_2_theta_15-15/reco_pdg_11_pt_2_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)


print("reco_pdg_11_pt_50_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_15-15/reco_pdg_11_pt_50_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)


print("reco_pdg_11_pt_5_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_5_theta_15-15/reco_pdg_11_pt_5_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)


print("reco_pdg_211_pt_100_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_100_theta_15-15/reco_pdg_211_pt_100_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_211_pt_10_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_10_theta_15-15/reco_pdg_211_pt_10_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_211_pt_150_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_150_theta_15-15/reco_pdg_211_pt_150_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_211_pt_1_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_1_theta_15-15/reco_pdg_211_pt_1_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_211_pt_200_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_200_theta_15-15/reco_pdg_211_pt_200_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_211_pt_2_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_2_theta_15-15/reco_pdg_211_pt_2_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_211_pt_50_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_15-15/reco_pdg_211_pt_50_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)

print("reco_pdg_211_pt_5_theta_15-15.root")
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_5_theta_15-15/reco_pdg_211_pt_5_theta_15-15.root")
events = file["events"]
clusters = events["PandoraClusters"]
print(events.num_entries)
