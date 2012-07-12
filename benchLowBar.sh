#! /bin/bash             
# generate the matrix

throttle=1
matrixSize=32

# small matrix sizes
for j in {1..42}
do
    ./genMatrix $matrixSize Inputs/matrixIn0.txt
    (/usr/bin/time -f "%e" ./lowbar) 2>> logs/log.txt
    matrixSize=$(($matrixSize+32))
done

# big matrix sizes
matrixSize=1376

for k in {1..11}
do
# gen the matrix
    ./genMatrix $matrixSize Inputs/matrixIn0.txt
# run the matrix multiply 32 times
    (/usr/bin/time -f "%e" ./lowbar) 2>> logs/log.txt
# double the matrix size
    matrixSize=$(($matrixSize+1376))
done

