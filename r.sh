#!/bin/bash

for i in 1 2 4 8 16 32
do
    ./b/reduction $i $((2*$i))
done

    ./b/reduction 31 31 
    ./b/reduction 63 63


