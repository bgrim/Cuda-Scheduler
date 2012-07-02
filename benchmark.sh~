#! /bin/bash                                                                                                                                                                       
throttle=1
# 1 through 5 for the throttles
for i in {1..5}
do
sleeptime=100
    # 1 through 8 for the sleeptimes
    for k in {1..8}
    do
	echo ""
	echo ""
	echo ""
	echo "Throttle count equals: " $throttle    
	echo "Sleeptime is equal to: " $sleeptime
	echo "The number of kernels: " $nkernels
	
	# run swift
	time ./run $throttle $sleeptime
	
        # double the sleeptime
	sleeptime=$(expr $sleeptime + $sleeptime)
    done
# double the throttle
# throttle=$(expr $throttle + 15)
# throttle=$(expr $throttle + 7)
# throttle=$(expr $throttle + 3)
#throttle=$(expr $throttle + 1)
throttle=$(expr $throttle + $throttle)
done