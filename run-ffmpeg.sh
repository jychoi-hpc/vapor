#!/bin/bash
# Example: run-ffmpeg.sh xgc_images-d3d_coarse_v2

DSET=$1

set -x
#ffmpeg -y -r 30 -f image2 -i $DSET/%06d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" $DSET.mp4
#ffmpeg -y -r 30 -f image2 -i $DSET/%06d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p -vf "crop=800:600" $DSET.mp4
ffmpeg -y -r 30 -f image2 -i $DSET/%06d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p $DSET.mp4

