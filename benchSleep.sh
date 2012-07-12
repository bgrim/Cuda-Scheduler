#! /bin/bash             
throttle=16
jobs=32
# do the overall test, this many times
for i in {1..1}
do
NOW=$(time +"%m-%d-%Y-%T")
sleepTime=100
    # 1 through 8 for the sleeptimes
    for k in {1..8}
    do
        for((c=0;c<$jobs;c++))
        do	
            # gen the file
	    ./genSleep $sleepTime Inputs/sleepIn$c.txt
	done

	# run the matrix multiply
	(/usr/bin/time -f "%e" ./run $throttle $jobs) 2>> logs/log$i.txt

	# double the matrix size
	sleepTime=$(($sleepTime+$sleepTime))
    done
done
