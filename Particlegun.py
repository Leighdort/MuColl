import argparse
import pyLCIO
import csv


X, Y, Z = 0, 1, 2
#Yippee this works
#Ok to tract one mc_particle I will use muon_2
#output_gen2
def options(): #needed when sending a file to something else 
    parser = argparse.ArgumentParser() #these are the things that are needed to call this command
    parser.add_argument("-i", required=True, help="Input LCIo file") #you need to give it a file
    parser.add_argument(
        "-n", required=False, type=int, help="Number of events to process" #optional to give it the max events
    )
    parser.add_argument("-o", required=True, help= "Output CSV File" ) #you need to give it an output file 
    return parser.parse_args()

def main():
    ops = options() #ok this means option is called first 
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader() #this is the reader to read the pyLCIO file
    reader.open(ops.i) #open the argument i, this is the name 
    all_events = []
  
    for i_event, event in enumerate(reader):
        if ops.n is not None and i_event >= ops.n:
            break #limits the number of events read 

        names = event.getCollectionNames()
        if "MCParticle" not in names:
            raise Exception(f"No MCParticle collection found in event {i_event}")
        mcparticles = event.getCollection("MCParticle")
        event_particles = []
        for mcparticle in mcparticles:
            momentum = mcparticle.getMomentum()
            vertex = mcparticle.getVertex()
            particle_info = { #creates a dictionary item of particle information
                "pdg": mcparticle.getPDG(),
                "event": i_event, 
                "energy": mcparticle.getEnergy(),
                "momentum": (momentum[X], momentum[Y], momentum[Z]),
                "vertex": (vertex[X], vertex[Y], vertex[Z]),
                "charge": mcparticle.getCharge(),
                "mass": mcparticle.getMass(),
                "time": mcparticle.getTime(),
                "generator_status": mcparticle.getGeneratorStatus()
            }
            event_particles.append(particle_info) #appends the mc particle 

        all_events.extend(event_particles)
        # Write to CSV
    with open(ops.o, "w", newline="") as csvfile: #has the output file, write mode, new line 
        fieldnames = [
            "event", "pdg", "energy", 
            "momentum",
            "vertex",
            "charge", "mass", "time", "generator_status"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for particle in all_events:
            writer.writerow(particle)

    print(f"Wrote {len(all_events)} particles to {ops.o}")

if __name__ == "__main__":
    main()