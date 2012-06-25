#! /bin/bash
for j in {1000..5000..1000}
do
for i in {1..10}
do
    time ./run $j
done
done