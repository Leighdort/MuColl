#Now we will check length
import uproot
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_11_nobib0.edm4hep.root")
events = file["events"]
print(events.num_entries)
