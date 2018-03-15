#!/bin/bash
for j in 36000 360000 3600000
do
for i in 1 2 4 8 16 32 64
do
    ./b/reduction $j 1 $((2*$i))
done
    ./b/reduction $j 32 32 
    ./b/reduction $j 64 64
    ./b/reduction $j 96 96
    ./b/reduction $j 128 128
done




