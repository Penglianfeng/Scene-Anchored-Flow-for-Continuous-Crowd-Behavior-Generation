#!/bin/bash
echo "Start downloading ETH/UCY dataset"
wget -O ewap_dataset_full.tgz https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz
wget -O crowd-data.zip --no-check-certificate https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
mkdir -p ETH-UCY/temp
tar -xvzf ewap_dataset_full.tgz -C ETH-UCY/temp
unzip crowd-data.zip -d ETH-UCY/temp
mkdir -p ETH-UCY/videos
mv ETH-UCY/temp/ewap_dataset/seq_eth/seq_eth.avi ETH-UCY/videos
mv ETH-UCY/temp/ewap_dataset/seq_hotel/seq_hotel.avi ETH-UCY/videos
mv ETH-UCY/temp/crowds/data/crowds_zara01.avi ETH-UCY/videos
mv ETH-UCY/temp/crowds/data/crowds_zara02.avi ETH-UCY/videos
mv ETH-UCY/temp/crowds/data/students003.avi ETH-UCY/videos
rm -rf ETH-UCY/temp
rm ewap_dataset_full.tgz
rm crowd-data.zip


