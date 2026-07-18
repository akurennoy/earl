#!/bin/sh
set -e
mkdir -p data
curl -sL -o data/amazon_mi.csv \
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Musical_Instruments.csv"
curl -sL -o data/ml-100k.zip \
  "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
unzip -o -q data/ml-100k.zip -d data
rm data/ml-100k.zip
echo "done: data/amazon_mi.csv data/ml-100k/"
