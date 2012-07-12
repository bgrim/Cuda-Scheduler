#! /bin/bash             
throttle=2
jobs=32
# do the overall test, this many times
for i in {1..1}
do
NOW=$(time +"%m-%d-%Y-%T")
matrixSize=32
    # 1 through 8 for the sleeptimes
    for k in {1..42}
    do
        for((c=0;c<$jobs;c++))
        do	
            # gen the file
	    ./genMatrix $matrixSize Inputs/matrixIn$c.txt
	done

	# run the matrix multiply
	(/usr/bin/time -f "%e" ./run $throttle $jobs) 2>> logs/log$i.txt

	# double the matrix size
	matrixSize=$(($matrixSize+32))
    done
    
matrixSize=1376
    # 1 through 8 for the sleeptimes
    for k in {1..8}
    do

	# generate the matrix
        for((c=0;c<=$jobs;c++))
        do
	  ./genMatrix $matrixSize Inputs/matrixIn$c.txt
        done

	# run the matrix multiply
	(/usr/bin/time -f "%e" ./run $throttle $jobs) 2>> logs/log$i.txt

	# double the matrix size
	matrixSize=$(($matrixSize+1376))    
    done
done
