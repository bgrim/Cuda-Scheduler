#! /bin/bash             
throttle=2
jobs=32
# do the overall test, this many times
for i in {1..1}
do
NOW=$(time +"%m-%d-%Y-%T")
matrixSize=1376
    # 1 through 8 for the sleeptimes
    for k in {1..10}
    do
	echo ""
	echo ""
	echo "Throttle count equals: " $throttle >> logs/log-$NOW.txt
	echo "The number of jobs is: " $jobs >> logs/log-$NOW.txt
	echo "The matrix size is   : " $matrixSize >> logs/log-$NOW.txt

	# generate the matrix
        for((c=0;c<=$jobs;c++))
        do
	  ./gen $matrixSize Inputs/matrixIn$c.txt
        done

	# run the matrix multiply
	(time ./run $throttle $jobs) 2>> logs/log-$NOW.txt

	# double the matrix size
	matrixSize=$(($matrixSize+1376))    
    done
done
