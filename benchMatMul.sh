#! /bin/bash                                                                                                                                                                       
throttle=1
jobs=1
# do the overall test, this many times
for i in {1..9}
do
matrixSize=1376
    # 1 through 8 for the sleeptimes
    for k in {1..10}
    do
	echo ""
	echo ""
	echo "Throttle count equals: " $throttle >> logs/log$i.txt
	echo "The number of jobs is: " $jobs >> logs/log$i.txt
	echo "The matrix size is   : " $matrixSize >> logs/log$i.txt

	# generate the matrix
	./gen $matrixSize Inputs/matrixIn0.txt

	# run the matrix multiply
	(time ./run $throttle $jobs) 2>> logs/log$i.txt

	# double the matrix size
	matrixSize=$(($matrixSize+1376))    
    done
done