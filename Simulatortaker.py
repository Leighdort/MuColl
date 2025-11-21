import argparse
import pyLCIO
import csv


X, Y, Z = 0, 1, 2

CAL_COLLECTIONS = [
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
]

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=str, help="Input LCIO file")
    parser.add_argument(
        "-n", required=False, type=int, help="Number of events to process"
    )
    parser.add_argument("-o", required=True, type=str, help="Output LCIO file")
    
    #I want to keep all of the particles 
    #parser.add_argument(
    #    "--nhits",
    #    default=10,
    #    type=int,
    #    help="Max number of hits to dump for each collection",
    #)
    return parser.parse_args()
def main():
    ops = options()
    print(f"Reading file{ops.i}")
    print(f"Printing hopefully all of the hits for each collection")
    all_events = []
    all_hits = []
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(ops.i)

    for i_event, event in enumerate(reader):
        if ops.n is not None and i_event >= ops.n:
            break
        cols = {}
        event_particles = []
        event_hits = []
        cols["MCParticle"] = get_collection(event, "MCParticle")
        for col in CAL_COLLECTIONS:
            cols[col] = get_collection(event, col)
        for i_mcparticle, mcparticle in enumerate(cols["MCParticle"]):
            #Should probably make a dictionary here 
            momentum = mcparticle.getMomentum()
            vertex = mcparticle.getVertex()
            particle_info = {
                "pdg": mcparticle.getPDG(),
                "event": i_event,
                "particle_num": i_mcparticle,
                "energy": mcparticle.getEnergy(),
                "momentum": (momentum[X], momentum[Y], momentum[Z]),
                "vertex": (vertex[X], vertex[Y], vertex[Z]),
                "charge": mcparticle.getCharge(),
                "mass": mcparticle.getMass(),
                "time": mcparticle.getTime(),
                "generator_status": mcparticle.getGeneratorStatus()
            }
            event_particles.append(particle_info) #appends the mc particle 
        for col_name in CAL_COLLECTIONS:
            for ihit, hit in enumerate(cols[col_name]):
                position = hit.getPosition()
                hit_info = {
                    "event": i_event,
                    "hit_num": ihit,
                    "type": col_name,
                    "energy": hit.getEnergy(),
                    "position": (
                        position[X],
                        position[Y],
                        position[Z]
                    )
                }
                event_hits.append(hit_info)
        all_events.extend(event_particles)
        all_hits.extend(event_hits)
# Write particles to CSV
    with open(ops.o.replace(".csv", "_particles.csv"), "w", newline="") as f_particles:
        particle_fieldnames = [
        "event", "particle_num", "pdg", "energy",
        "momentum", "vertex", "charge", "mass",
        "time", "generator_status"
        ]
        writer = csv.DictWriter(f_particles, fieldnames=particle_fieldnames)
        writer.writeheader()
        for particle in all_events:
            writer.writerow(particle)

    # Write hits to CSV
    with open(ops.o.replace(".csv", "_hits.csv"), "w", newline="") as f_hits:
        hit_fieldnames = [
            "event", "hit_num", "type", "energy", "position"
        ]
        writer = csv.DictWriter(f_hits, fieldnames=hit_fieldnames)
        writer.writeheader()
        for hit in all_hits:
            writer.writerow(hit)

    print(f"Wrote {len(all_events)} particles and {len(all_hits)} hits to CSV.")


def get_collection(event, name):
    names = event.getCollectionNames()
    if name in names:
        return event.getCollection(name)
    return []
if __name__ == "__main__":
    main()
