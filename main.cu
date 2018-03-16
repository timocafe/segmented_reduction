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

void init_cpu(std::vector<float> &v_data, std::vector<int> &v_offset, std::vector<int> &v_size, int min, int max){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(min, max); // 64 max element
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
//  std::fill(v_data.begin(),v_data.end(),1);
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



__global__ void kernel_gpu_original( const float* __restrict__  p_data,const int* __restrict__ p_offset, float* __restrict__ p_res, int size_data, int size_res){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int ir  = tid; ir < size_res; ir += blockDim.x * gridDim.x) {
            auto begin = p_offset[ir];
            auto end =  p_offset[ir+1];
            for(auto j = begin; j < end; ++j)
                p_res[ir] += p_data[j];
    }
}
enum reduce {partial = 2, full = 32};


//reduce a single hardware warp;
template<class T>
__inline__ __device__ T warpReduceSum(T val, reduce r){
    #pragma unroll
    for(int i = 1; i < r; i *= 2)
        val += __shfl_xor_sync(0xffffffff,val, i, 32);
    return val;
}


__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}


template<int N>
__global__ void kernel_gpu_tune( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res);
/*
template<>
__global__ void kernel_gpu_tune<32>( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x);
    const int size_buffer_shared_memory = 16384;
    __shared__ float sdata[size_buffer_shared_memory]; // 256 * 64

    int n_tid = tid; // first iteration we start at 0
   
    const int block_by_grid_dim = blockDim.x * gridDim.x;

    for (int in = tid; in  < 32*size_res; in += block_by_grid_dim) {
    
        const int warpid = in/32;
        const int laneid = lane_id();

        //get the value from the offset only the thread 0 (lane id) of the warp
        const int offset_value = p_offset[warpid];
        const int size_receptor = p_size[warpid];
        const int offset_id = (laneid + offset_value);

        sdata[tid] = p_data[n_tid];
        __syncthreads();   
 
        data_for_reduction = (laneid < size_receptor) ? s_data[offset_id] : 0;

        auto r = warpReduceSum(data_for_reduction,full); // r all the same value for a warp

        p_res[warpid] = r; 
        n_tid = ????;
    }
}
*/
template<>
__global__ void kernel_gpu_tune<32>( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x);
    const int block_by_grid_dim = blockDim.x * gridDim.x;

    for (int in = tid; in  < 32*size_res; in += block_by_grid_dim) {
    
        const int warpid = in/32;
        const int laneid = lane_id();

        //get the value from the offset only the thread 0 (lane id) of the warp
        const int offset_value = p_offset[warpid];
        const int size_receptor = p_size[warpid];
        const int offset_id = (laneid + offset_value);

        data_for_reduction = (laneid < size_receptor) ? p_data[offset_id] : 0;

        auto r = warpReduceSum(data_for_reduction,full); // r all the same value for a warp

        p_res[warpid] = r; 
    }
}

template<>
__global__ void kernel_gpu_tune<64>( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x);

    for (int in = tid; in  < 32*size_res; in += (blockDim.x * gridDim.x)) {
    
        const int warpid = in/32;
        const int laneid = lane_id();

        //get the value from the offset only the thread 0 (lane id) of the warp
        const int offset_value = p_offset[warpid];
        const int size_receptor = p_size[warpid];
     
        const int offset_id = laneid + offset_value;

        data_for_reduction = (laneid < size_receptor) ? p_data[offset_id] : 0;
        data_for_reduction += (laneid + 32 < size_receptor) ? p_data[offset_id + 32] : 0;

        auto r = warpReduceSum(data_for_reduction,full); // r all the same value for a warp

        p_res[warpid] = r; 
    }
}

template<>
__global__ void kernel_gpu_tune<96>( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x);

    for (int in = tid; in  < 32*size_res; in += (blockDim.x * gridDim.x)) {
    
        const int warpid = in/32;
        const int laneid = lane_id();

        //get the value from the offset only the thread 0 (lane id) of the warp
        const int offset_value = p_offset[warpid];
        const int size_receptor = p_size[warpid];
     
        const int offset_id = laneid + offset_value;

        data_for_reduction =  (laneid < size_receptor) ? p_data[offset_id] : 0;
        data_for_reduction += (laneid + 32 < size_receptor) ? p_data[offset_id + 32] : 0;
        data_for_reduction += (laneid + 64 < size_receptor) ? p_data[offset_id + 64] : 0;

        auto r = warpReduceSum(data_for_reduction,full); // r all the same value for a warp

        p_res[warpid] = r; 
    }
}

template<>
__global__ void kernel_gpu_tune<128>( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x);

    for (int in = tid; in  < 32*size_res; in += (blockDim.x * gridDim.x)) {
    
        const int warpid = in/32;
        const int laneid = lane_id();

        //get the value from the offset only the thread 0 (lane id) of the warp
        const int offset_value = p_offset[warpid];
        const int size_receptor = p_size[warpid];
     
        const int offset_id = laneid + offset_value;

        data_for_reduction = (laneid < offset_value+size_receptor) ? p_data[offset_id] : 0;
        data_for_reduction += (laneid + 32 < size_receptor) ? p_data[offset_id + 32] : 0;
        data_for_reduction += (laneid + 64 < size_receptor) ? p_data[offset_id + 64] : 0;
        data_for_reduction += (laneid + 96 < size_receptor) ? p_data[offset_id + 96] : 0;

        auto r = warpReduceSum(data_for_reduction,full); // r all the same value for a warp

        p_res[warpid] = r; 
    }
}

int main(int argc, const char * argv[]) {
    int size = atoi(argv[1]);
    int thread = 256; // atoi(argv[2]);
    int block = 256; // atoi(argv[3]);
    int min = atoi(argv[2]);
    int max = atoi(argv[3]);
    // cpu
    std::vector<int> v_offset(size);
    std::vector<int> v_size(size-1);
    std::vector<float> v_data;
    std::vector<float> v_res(size-1);
    std::vector<float> v_res_gpu(size-1);
    std::vector<float> v_res_gpu2(size-1);
    init_cpu(v_data,v_offset,v_size, min, max);

    //gpu
    int * p_offset;
    int * p_size;
    float* p_data;
    float* p_res;


   //  std::cout << " size data " << v_data.size() << " \n";

    cudaMalloc((void**)&p_offset, v_offset.size()*sizeof(int));
    cudaMalloc((void**)&p_size, v_offset.size()*sizeof(int));
    cudaMalloc((void**)&p_data, v_data.size()*sizeof(float));
    cudaMalloc((void**)&p_res, v_res.size()*sizeof(float));

    cudaMemcpy(p_offset,&v_offset[0],v_offset.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(p_size,&v_size[0],v_size.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(p_data,&v_data[0],v_data.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(p_res,&v_res_gpu[0],v_res_gpu.size()*sizeof(float),cudaMemcpyHostToDevice);

    auto start =    std::chrono::high_resolution_clock::now();
    kernel_cpu(v_data,v_offset,v_res);
    auto end =   std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    cudaEvent_t start_original, stop_original;
    cudaEvent_t start_64, stop_64;
    cudaEventCreate(&start_original);
    cudaEventCreate(&start_64);
    cudaEventCreate(&stop_original);
    cudaEventCreate(&stop_64);

    cudaEventRecord(start_original);
    kernel_gpu_original<<<block,thread>>>(p_data,p_offset,p_res,v_data.size(),v_res.size());
    cudaEventRecord(stop_original); 
 
    cudaMemcpy(&v_res_gpu[0],p_res,v_res_gpu.size()*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemset(p_res,0,v_res_gpu.size()*sizeof(float));


    cudaEventRecord(start_64);
    if(max <= 32)
        kernel_gpu_tune<32><<<block,thread>>>(p_data,p_offset, p_size,p_res,v_data.size(),v_res.size());
    else if ( max <= 64)
        kernel_gpu_tune<64><<<block,thread>>>(p_data,p_offset, p_size,p_res,v_data.size(),v_res.size());
    else if ( max <= 96)
        kernel_gpu_tune<96><<<block,thread>>>(p_data,p_offset, p_size,p_res,v_data.size(),v_res.size());
    else 
        kernel_gpu_tune<128><<<block,thread>>>(p_data,p_offset, p_size,p_res,v_data.size(),v_res.size());
    cudaEventRecord(stop_64); 

    cudaMemcpy(&v_res_gpu2[0],p_res,v_res_gpu2.size()*sizeof(float),cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_original);
    cudaEventSynchronize(stop_64);
  
   float milliseconds_original = 0;
   float milliseconds_64 = 0;

   cudaEventElapsedTime(&milliseconds_original, start_original, stop_original);
   cudaEventElapsedTime(&milliseconds_64, start_64, stop_64);

    auto sum_cpu = std::accumulate(v_res.begin(), v_res.end(), 0.);
    auto sum_gpu_original = std::accumulate(v_res_gpu.begin(), v_res_gpu.end(), 0.);
    auto sum_gpu_tune = std::accumulate(v_res_gpu2.begin(), v_res_gpu2.end(), 0.);

//  std::cout << " memory allocated : data " << v_data.size()*sizeof(float)/1048576. << " [mB]\n ";
//  std::cout << " memory allocated : offset " << v_offset.size()*sizeof(float)/1048576. << " [mB]\n ";
//  std::cout << " memory allocated : size  " << v_size.size()*sizeof(float)/1048576. << " [mB]\n ";
//  std::cout << " memory allocated : res  " << v_res_gpu.size()*sizeof(float)/1048576. << " [mB]\n ";
//  std::cout << " sum cpu " << sum_cpu << " sum gpu original " << sum_gpu_original << " sum gpu tune " << sum_gpu_tune << std::endl;

//  for(int i = 0 ; i < size-1; ++i)
//     std::cout << " reduction: "<< i << " range ["  << v_offset[i] <<","<< v_offset[i+1]  << "], cpu:" << v_res[i]  << ", gpu:" << v_res_gpu[i] << ", gpu2:" << v_res_gpu2[i]<< std::endl;

    // insert code here...
//  std::cout << " min " << min << " max " << max << " time cpu:" << elapsed_seconds.count()*1000 << " [ms], gpu original " << milliseconds_original << " [ms], gpu tune " << milliseconds_64  << " [ms] \n ";
    std::cout << size << "," << min << "," << max << "," <<  elapsed_seconds.count()*1000 << "," << milliseconds_original << "," << milliseconds_64  << "\n ";
    return 0;
}
