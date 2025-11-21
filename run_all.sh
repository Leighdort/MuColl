#This is running many files at once


#!/bin/bash
energies=(10 50 100 150 200)
#energies=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
#energies=(10 20)
#i=10
#first do a practice one to see what gets made
#Charged electron
for E in "${energies[@]}"; do
    echo ">>> Running with beam energy ${E} GeV"
    python mucoll-benchmarks/generation/pgun/pgun_lcio.py \
    -s 12345 \
    -e 1000 \
    --pdg 211 \
    --p $E \
    --theta 15 \
    -- outputpion_gen${E}.slcio
    ddsim \
    --steeringFile mucoll-benchmarks/simulation/ilcsoft/steer_baseline.py \
    --inputFile outputpion_gen${E}.slcio \
    --outputFile outputpion_sim${E}.slcio
    k4run mucoll-benchmarks/digitisation/k4run/digi_steer.py --LcioEvent.Files outputpion_sim${i}.slcio
     #   -> yes there is a way to rename files. This should be done to keep all the files and then all my
     # like reading root files
    mv output_digi.slcio outputpion_digi${E}.slcio
    cp -a mucoll-benchmarks/reconstruction/k4run/PandoraSettings ./
    k4run mucoll-benchmarks/reconstruction/k4run/reco_steer.py \
    --LcioEvent.Files outputpion_digi${E}.slcio \
    --MatFile ${ACTS_MatFile} \
    --TGeoFile ${ACTS_TGeoFile} 
    mv output_reco.slcio outputpion_reco${E}.slcio
    python Recontaker.py -i outputpion_reco${E}.slcio -o outputpion_reco${E}.csv
    #i=$((i+10))
    #Now call the one that gets clusters
done
