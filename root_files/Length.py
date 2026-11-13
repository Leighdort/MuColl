import uproot

files = {
    11: "/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_11_nobib0.edm4hep.root",
    211: "/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root",
    2112: "/users/rldohert/data/mucoll/rldohert/pdg_2112_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_2112_nobib0.edm4hep.root",
}

for pdg, path in files.items():
    file = uproot.open(path)
    events = file["events"]
        count = 0

    for i in range(events.num_entries):
        count += 1

    print(count)
    print(f"PDG {pdg}: {len(events)} events")