#! /bin/bash                                                                                                                                                                       
throttle=1
jobs=1
# 1 through 5 for the throttles
for i in {1..10}
do
matrixSize=32
    # 1 through 8 for the sleeptimes
    for k in {1..12}
    do
	echo ""
	echo ""
	echo "Throttle count equals: " $throttle >> logs/log$i.txt
	echo "The number of jobs is: " $jobs >> logs/log$i.txt
	echo "The matrix size is   : " $matrixSize >> logs/log$i.txt

	# generate the matrix
	./gen $matrixSize

	# run the matrix multiply
	(time ./run $throttle $jobs) 2>> logs/log$i.txt

	# double the matrix size
	matrixSize=$(($matrixSize+$matrixSize))    
    done
done