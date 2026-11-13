
import uproot

f = uproot.open("/users/rldohert/data/mucoll/rldohert/pdg_211_pt_5_theta_15-15/digi_pdg_211_pt_5_theta_15-15.root")

print(f["events"].keys())

#Let's examine the digi here