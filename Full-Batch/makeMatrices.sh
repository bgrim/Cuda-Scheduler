#! /bin/bash
sideLength=320

for i in {0..63}
do
  ./gen $sideLength "Inputs/matrixIn${i}.txt"
done