#! /bin/bash
sideLength=4096

for i in {0..15}
do
  ./gen $sideLength "Inputs/matrixIn${i}.txt"
done