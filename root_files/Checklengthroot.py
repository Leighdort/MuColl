import uproot

# List of your files
#energies = [10, 50, 100, 150, 200]
#energies = [10, 50, 100]
#for num in energies:
#    filename = f"reco_output.edm4hep.root"
#    try:
#        file = uproot.open(filename)
#        events = file["events"]
#        n_events = events.num_entries
#        print(f"{filename}: {n_events} events")
#    except Exception as e:
#        print(f"Error opening {filename}: {e}")
#   
filename = f"reco_output.edm4hep.root"
file = uproot.open(filename)
events = file["events"]
n_events = events.num_entries
print(f"{filename}: {n_events} events")
filename = f"reco_outputp2.10.edm4hep.root"
file = uproot.open(filename)
events = file["events"]
n_events = events.num_entries
print(f"{filename}: {n_events} events")
filename = f"reco_outpute10.edm4hep.root"
file = uproot.open(filename)
events = file["events"]
n_events = events.num_entries
print(f"{filename}: {n_events} events")