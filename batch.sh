#! /bin/bash
throttle=1
for i in {1..5}
do
   sleepTime=100
   for j in {1..7}
   do
        time ./batch $throttle 64 $sleepTime
        let "sleepTime<<=1"
   done
   let "throttle<<=1"
done