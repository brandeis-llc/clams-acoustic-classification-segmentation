#!/bin/bash 

# important! INDIR should not have any subdirs and all audio files need to be placed immediatedly under
INDIR=$(realpath $1)

# change this to your docker image names
AES_IMAGE="keigh-aes"
KALDI_IMAGE="aapb-pua-kaldi"

# audio segmentation 
docker run --rm -v $INDIR:/segmenter/data $AES_IMAGE

for splitdir in $(find $INDIR -maxdepth 1 -type d); do 
    if [ $splitdir != $INDIR ]; then 
        # call kaldi on each segment
        docker run --rm -v "$splitdir":/audio_in $KALDI_IMAGE
        # merge segmental transcripts into one json
        # merger script is provided via aes docker image
        docker run --rm -v "$INDIR":/segmenter/data $AES_IMAGE python /segmenter/merge_jsons.py /segmenter/data/$(basename "$splitdir")
    fi
done
