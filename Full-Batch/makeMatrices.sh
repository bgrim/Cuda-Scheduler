#! /bin/bash
sideLength=96

for i in {0..63}
do
  ./gen $sideLength "Inputs/matrixIn${i}.txt"
done