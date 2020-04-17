#!/bin/bash 

# important! INDIR should not have any subdirs and all audio files need to be placed immediatedly under
INDIR=$(realpath $1)

# change this to your docker image names
AES_IMAGE="keigh-aes"
KALDI_IMAGE="aapb-pua-kaldi"

docker run --rm -v $INDIR:/segmenter/data $AES_IMAGE

for splitdir in $(find $INDIR -type d); do 
    if [ $splitdir != $INDIR ]; then 
        docker run --rm -v $splitdir:/audio_in $KALDI_IMAGE
        python merge_jsons.py $splitdir
    fi
done
