# Checking:
# PFO -> AllTracks
# SiTracks -> AllTracks
#
# For every PFO with an AllTracks association, print:
#   PFO index
#   AllTracks index
#   AllTracks collection ID
#   corresponding SiTracks index, if found

import numpy as np
import uproot

choices = [50]

matches = 0
total = 0
for num in choices:

    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root")

    events = file["events"]

    # ---------------------------------------------------------
    # Collection ID for AllTracks
    # ---------------------------------------------------------

    ALLTRACKS_ID = 4111979324

    # ---------------------------------------------------------
    # PFO -> Track relations
    # ---------------------------------------------------------

    pandora_tracks_begin = events[
        "PandoraPFOs/PandoraPFOs.tracks_begin"
    ].array()

    pandora_tracks_end = events[
        "PandoraPFOs/PandoraPFOs.tracks_end"
    ].array()

    pandora_tracks_index = events[
        "_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"
    ].array()

    pandora_tracks_collection = events[
        "_PandoraPFOs_tracks/_PandoraPFOs_tracks.collectionID"
    ].array()

    # ---------------------------------------------------------
    # SiTracks -> original collection information
    # ---------------------------------------------------------

    si_tracks_index = events[
        "SiTracks_objIdx/SiTracks_objIdx.index"
    ].array()

    si_tracks_collection = events[
        "SiTracks_objIdx/SiTracks_objIdx.collectionID"
    ].array()

    # ---------------------------------------------------------
    # Build:
    #
    # AllTracks index -> SiTracks index
    #
    # This is useful because Pandora gives us the AllTracks ID,
    # while SiTracks_objIdx tells us which AllTracks object
    # each SiTrack came from.
    # ---------------------------------------------------------

    alltrack_to_sitrack = {}

    for i in range(events.num_entries):

        for si_index, collection_id in zip(
            si_tracks_index[i],
            si_tracks_collection[i]
        ):

            si_index = int(si_index)
            collection_id = int(collection_id)

            if collection_id == ALLTRACKS_ID:

                # SiTracks[si] points to AllTracks[si_index]
                key = (i, si_index)

                alltrack_to_sitrack[key] = si_index

    # ---------------------------------------------------------
    # Loop over events
    # ---------------------------------------------------------

    total_pfos = 0
    pfos_with_alltrack = 0

    print("\n==============================================")
    print("PFO -> AllTracks association")
    print("==============================================")

    for i in range(events.num_entries):

        print(f"\n================ Event {i} ================")

        num_pfos = len(
            pandora_tracks_begin[i]
        )

        total_pfos += num_pfos

        for pfo in range(num_pfos):

            track_begin = int(
                pandora_tracks_begin[i][pfo]
            )

            track_end = int(
                pandora_tracks_end[i][pfo]
            )

            print(f"\nPFO {pfo}")
            print(
                f"  Track relation range: "
                f"{track_begin} -> {track_end}"
            )

            found_alltrack = False

            for j in range(track_begin, track_end):

                track_index = int(
                    pandora_tracks_index[i][j]
                )

                collection_id = int(
                    pandora_tracks_collection[i][j]
                )

                # -------------------------------------------------
                # Pandora is using AllTracks
                # -------------------------------------------------

                if collection_id == ALLTRACKS_ID:
                    total + = 1
                    found_alltrack = True
                    pfos_with_alltrack += 1

                    print("  --------------------------------")
                    print("  Pandora-associated track:")
                    print(
                        f"    AllTracks index = {track_index}"
                    )
                    print(
                        f"    AllTracks collectionID = "
                        f"{collection_id}"
                    )

                    # -------------------------------------------------
                    # Does this AllTracks track appear in SiTracks?
                    # -------------------------------------------------

                    si_match = None

                    for si_index, si_collection_id in zip(
                        si_tracks_index[i],
                        si_tracks_collection[i]
                    ):

                        si_index = int(si_index)
                        si_collection_id = int(
                            si_collection_id
                        )

                        if (
                            si_collection_id == ALLTRACKS_ID
                            and si_index == track_index
                        ):
                            si_match = si_index
                            break

                    if si_match is not None:

                        print(
                            f"    SiTracks association = "
                            f"SiTracks[{si_match}]"
                        )

                    else:

                        print(
                            "    SiTracks association = "
                            "NONE"
                        )
                    if track_index == si_match:
                        matches += 1

            if not found_alltrack:

                print(
                    "  No AllTracks associated with this PFO"
                )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n==============================================")
    print("FINAL RESULTS")
    print("==============================================")

    print(
        f"Total PFOs: "
        f"{total_pfos}"
    )

    print(
        f"PFOs with an AllTracks association: "
        f"{pfos_with_alltrack}"
    )

    if total_pfos > 0:

        print(
            f"Fraction of PFOs with AllTracks: "
            f"{pfos_with_alltrack / total_pfos:.4f}"
        )

        print(
            f"Percentage: "
            f"{100 * pfos_with_alltrack / total_pfos:.2f}%"
        )
print(f"total: {total}")
print(f"matches: {matches}")