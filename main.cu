//
//  main.cpp
//  reduction
//
//  Created by Timothée Ewart on 06/03/2018.
//  Copyright © 2018 Timothée Ewart. All rights reserved.
//

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <cassert>

void init_cpu(std::vector<float> &v_data, std::vector<int> &v_offset, std::vector<int> &v_size){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(1, 63); // 64 max element
    std::uniform_real_distribution<> disf(1, 10); // 100 max element
    auto rand = std::bind(dis, gen);
    auto randf = std::bind(disf, gen);
    // receptor ok
    std::generate(v_offset.begin()+1,v_offset.end(),[&](){return rand();});
    std::copy(v_offset.begin()+1,v_offset.end(),v_size.begin());
    auto sum = std::accumulate(v_offset.begin(), v_offset.end(), 0);
    v_data.resize(sum);
    // the data
    std::generate(v_data.begin(),v_data.end(),[&](){return randf();});
//  std::fill(v_data.begin(),v_data.end(),1.1);
//  std::iota(v_data.begin(),v_data.end(),0);



    int end = v_offset[0];
    for(int i(0); i < v_offset.size(); ++i){
        v_offset[i] = end;
        end += v_offset[i+1]; // overflow ....
    }
}

void kernel_cpu(std::vector<float> &v_data, std::vector<int> &v_offset, std::vector<float> &v_res){
    auto size = v_data.size();
    for(int i(0); i < v_offset.size()-1; ++i){
        auto begin = v_offset[i];
        auto end = v_offset[i+1];
        for(int j = begin; j < end; ++j)
            v_res[i] += v_data[j];
    }
}



__global__ void kernel_gpu_original(float* p_data, int* p_offset, float* p_res, int size_data, int size_res){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for (int ir  = tid; ir < size_res; ir += blockDim.x * gridDim.x) {
        if(ir < size_res){
            auto begin = p_offset[ir];
            auto end = p_offset[ir+1];
            for(int j = begin; j < end; ++j)
                p_res[ir] += p_data[j];
        }
    }
}


template<int N>
__forceinline__ __device__ unsigned lane_id();

template<>
__forceinline__ __device__ unsigned lane_id<32>(){
 // unsigned ret;
 // asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
 // return ret;
    return threadIdx.x % 32;
}

template<>
__forceinline__ __device__ unsigned lane_id<64>(){
    return threadIdx.x % 64;
}


__forceinline__ __device__ unsigned warp_id(int size_swarp){
    return threadIdx.x / size_swarp;
}


enum reduce {partial = 2, full = 32};

//reduce a single hardware warp;
template<class T>
__inline__ __device__ T warpReduceSum(T val, reduce r){
    #pragma unroll
    for(int i = 1; i < r; i *= 2)
        val += __shfl_xor(val, i);
    return val;
}

//reduce several warp over a single block i.e. <1,64> : we have two warps
template<class T>
__device__ T blockReduceSum(T val){
    int warpid = warp_id(32); //number of the warp depend of the number of thread i.e. #threads/32
    int laneid = lane_id<32>(); //threadId into the warp [0,32]

    __shared__ T shared[32]; //32 because hardware size
    if(threadIdx.x < 32) shared[threadIdx.x]= 0;
    __syncthreads();

    val = warpReduceSum(val,full); // each warp is doing partial reduction

    if(laneid==0) shared[warpid] = val; // Write reduce sum in shared mem.
    __syncthreads();

    //read from shared mem only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : 0;
    __syncthreads();
    val = warpReduceSum(val,partial); // if 64 partial
    val = __shfl(val,threadIdx.x*2); // A B C D E F G H becomes A C D F ...
    return val;
}

__global__ void kernel_gpu_64(float* p_data, int* p_offset, int* p_size, float* p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    int global_blockDim = blockIdx.x*blockDim.x;
    int tid = (blockIdx.x*blockDim.x+threadIdx.x);
    #pragma unroll
    for (int in = tid; in/64 < size_res; in += (blockDim.x * gridDim.x)) {
 
        int global_warpid_64 = in/64;
        int global_laneid_32 = in%32;
        int global_laneid_64 = in%64;
        
        //value for the size of each receptor
        int size_receptor;
        // value for the offset
        int offset_value;
        //get the value from the offset only the thread 0 (lane id) of the warp
        if((global_laneid_32 == 0) && (global_warpid_64 < size_res)){
            offset_value = p_offset[global_warpid_64];
            size_receptor = p_size[global_warpid_64];
        }
        
        //broadcast the value from lane id 0 to all lane id of the warp
        offset_value = __shfl(offset_value,0); 
        size_receptor =  __shfl(size_receptor,0);
        //get the correct indices with the offset and the lane id (NOT tid)
        int offset_id = global_laneid_64 + offset_value;
        
        // branching to avoid overflow over all the data
        if(offset_id < size_data ){
            data_for_reduction = p_data[offset_id];
            if(tid%64 >= size_receptor){
                data_for_reduction = 0;
            }
        }
        
        auto tmp = blockReduceSum(data_for_reduction);
       
        if(threadIdx.x < blockDim.x/64){
            int tid_final = (global_blockDim/64+threadIdx.x);
            p_res[tid_final] = tmp; // it will sum 0
        }
        global_blockDim += blockDim.x * gridDim.x; 
    }
}


int main(int argc, const char * argv[]) {
    int size = atoi(argv[1]);
    int thread = atoi(argv[2]);
    int block = atoi(argv[3]);
    // cpu
    std::vector<int> v_offset(size);
    std::vector<int> v_size(size-1);
    std::vector<float> v_data;
    std::vector<float> v_res(size-1);
    std::vector<float> v_res_gpu(size-1);
    std::vector<float> v_res_gpu2(size-1);
    init_cpu(v_data,v_offset,v_size);


    //gpu
    int * p_offset;
    int * p_size;
    float* p_data;
    float* p_res;


       std::cout << " size data " << v_data.size() << " \n";

    cudaMalloc((void**)&p_offset, v_offset.size()*sizeof(int));
    cudaMalloc((void**)&p_size, v_offset.size()*sizeof(int));
    cudaMalloc((void**)&p_data, v_data.size()*sizeof(float));
    cudaMalloc((void**)&p_res, v_res.size()*sizeof(float));

    cudaMemcpy(p_offset,&v_offset[0],v_offset.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(p_size,&v_size[0],v_size.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(p_data,&v_data[0],v_data.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(p_res,&v_res_gpu[0],v_res_gpu.size()*sizeof(float),cudaMemcpyHostToDevice);

    auto start =  std::chrono::high_resolution_clock::now();
    kernel_cpu(v_data,v_offset,v_res);
    auto end =  std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    kernel_gpu_original<<<block,thread>>>(p_data,p_offset,p_res,v_data.size(),v_res.size());
    cudaMemcpy(&v_res_gpu[0],p_res,v_res_gpu.size()*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemset(p_res,0,v_res_gpu.size()*sizeof(float));
    kernel_gpu_64<<<block,thread>>>(p_data,p_offset, p_size,p_res,v_data.size(),v_res.size());
    cudaMemcpy(&v_res_gpu2[0],p_res,v_res_gpu2.size()*sizeof(float),cudaMemcpyDeviceToHost);

    auto sum_cpu = std::accumulate(v_res.begin(), v_res.end(), 0.);
    auto sum_gpu_original = std::accumulate(v_res_gpu.begin(), v_res_gpu.end(), 0.);
    auto sum_gpu_tune = std::accumulate(v_res_gpu2.begin(), v_res_gpu2.end(), 0.);

    std::cout  << " sum cpu " << sum_cpu << " sum gpu original " << sum_gpu_original << " sum gpu tune " << sum_gpu_tune << std::endl;
 //for(int i = 0 ; i < size-1; ++i)
//     std::cout << " reduction: "<< i << " range ["  << v_offset[i] <<","<< v_offset[i+1]  << "], cpu:" << v_res[i]  << ", gpu:" << v_res_gpu[i] << ", gpu2:" << v_res_gpu2[i]<< std::endl;

    // insert code here...
    std::cout << "time " << elapsed_seconds.count() << " [s] \n";
    return 0;
}
