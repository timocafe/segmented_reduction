# What is it ?
This tiny program make a large number of tiny reduction from a contiguos buffer. 
The number of value for reduction are between 1 and 63

# Algorithms
1 The classical one straightforward 
2 The Nvidia one inspired from the shuffle reduction

# Compilation
nvcc -O3 --std=c++11 --gpu-architecture sm_50 -Xptxas="-v"   main.cu 

# Execution
./a.out #number_of_reduction #threads.x #block.x

