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
    std::generate(v_offset.begin()+1,v_offset.end(),[&](){return  rand();});
    std::copy(v_offset.begin()+1,v_offset.end(),v_size.begin());
    auto sum = std::accumulate(v_offset.begin(), v_offset.end(), 0);
    v_data.resize(sum);
    // the data
    std::generate(v_data.begin(),v_data.end(),[&](){return randf();});
    std::fill(v_data.begin(),v_data.end(),1);
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

//reduce several warp over a single block i.e. <1,64> : we have two warps
template<class T, int N>
struct blockReduceSumHelper;

//partial for number belong [0,64[
template<class T>
struct blockReduceSumHelper<T,64>{
static __inline__ __device__ T blockReduceSum(T val){
        int warpid = threadIdx.x/32; //number of the warp depend of the number of thread i.e. #threads/32
        int laneid = threadIdx.x%32; //threadId into the warp [0,32]

        __shared__ T shared[32]; //32 because hardware size

        val = warpReduceSum(val,full); // each warp is doing partial reduction
    
        if(laneid==0) shared[warpid] = val; // Write reduce sum in shared mem.
        __syncthreads();

        //read from shared mem only if that warp existed
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : 0;
        val = warpReduceSum(val,partial); // if 64 partial
        val = __shfl_sync(0xffffffff,val,threadIdx.x*2, 32); // A B C D E F G H becomes A C D F ...

        return val;
    }
};

//partial for number belong [0,31[
template<class T>
struct blockReduceSumHelper<T,32>{
static __inline__ __device__ T blockReduceSum(T val){
    int warpid = threadIdx.x/32; //number of the warp depend of the number of thread i.e. #threads/32
    int laneid = threadIdx.x%32; //threadId into the warp [0,32]

    __shared__ T shared[32]; //32 because hardware size
    val = warpReduceSum(val,full); // each warp is doing partial reduction
    if(laneid==0) shared[warpid] = val; // Write reduce sum in shared mem.
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : 0;
//  printf("line, val %d %f \n", laneid, shared[laneid]);
    return shared[laneid];
    }
};

__global__ void kernel_gpu_tune( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    int global_blockDim = blockIdx.x*blockDim.x;
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x);

    const int max_reduction_size = 64;
    const int max_threads_per_warps = 32;
 
    for (int in = tid; in/max_reduction_size  < size_res; in += (blockDim.x * gridDim.x)) {
    
        const int warpid = in/max_reduction_size;
        const int laneid = in%max_reduction_size;

        //get the value from the offset only the thread 0 (lane id) of the warp
        const int offset_value = p_offset[warpid];
        const int size_receptor = p_size[warpid];
     
        int offset_id = laneid + offset_value;
        #pragma unroll  
        for (int j = 0; j < 2  ; ++j){
            (offset_id < size_data && tid%max_reduction_size < size_receptor ) ? data_for_reduction += p_data[offset_id] : 0;
            offset_id += max_threads_per_warps;
        }

        printf("data %f: \n",data_for_reduction); 
 
        auto tmp =  blockReduceSumHelper<float,32>::blockReduceSum(data_for_reduction);
      
    
        printf("tmp %f: \n",tmp); 
 
 //     if(threadIdx.x < blockDim.x/64){
           int tid_final = (global_blockDim/64+threadIdx.x);
           printf(" td_final %d %d \n", tid_final, threadIdx.x);
            if(tid_final < size_res)
                p_res[tid_final] = tmp; // it will sum 0
//      }
        global_blockDim += blockDim.x * gridDim.x; 
        data_for_reduction = 0;
    }
}

template<int N>
__global__ void kernel_gpu( const float* __restrict__ p_data, const int* __restrict__ p_offset, const int* __restrict__ p_size, float* __restrict__ p_res, int size_data, int size_res){
    float data_for_reduction = 0;
    int global_blockDim = blockIdx.x*blockDim.x;
    const int tid = (blockIdx.x*blockDim.x+threadIdx.x);

 
    for (int in = tid; in/N < size_res; in += (blockDim.x * gridDim.x)) {
    
        const int global_warpid_N = in/N;
        const int global_laneid_N = in%N;
        
        //get the value from the offset only the thread 0 (lane id) of the warp
        const int offset_value = p_offset[global_warpid_N];
        const int size_receptor = p_size[global_warpid_N];
        
        //get the correct indices with the offset and the lane id (NOT tid)
        int offset_id = global_laneid_N + offset_value;
    
        // branching to avoid overflow over all the data
        (offset_id < size_data && tid%N < size_receptor ) ? data_for_reduction = p_data[offset_id] : data_for_reduction = 0;
        
        auto tmp =  blockReduceSumHelper<float,N>::blockReduceSum(data_for_reduction);
       
        if(threadIdx.x < blockDim.x/N){
            int tid_final = (global_blockDim/N+threadIdx.x);
            if(tid_final < size_res)
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
  //kernel_gpu<64><<<block,thread>>>(p_data,p_offset, p_size,p_res,v_data.size(),v_res.size());
    kernel_gpu_tune<<<block,thread>>>(p_data,p_offset, p_size,p_res,v_data.size(),v_res.size());
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

    std::cout << " memory allocated : data " << v_data.size()*sizeof(float)/1048576. << " [mB]\n ";
    std::cout << " memory allocated : offset " << v_offset.size()*sizeof(float)/1048576. << " [mB]\n ";
    std::cout << " memory allocated : size  " << v_size.size()*sizeof(float)/1048576. << " [mB]\n ";
    std::cout << " memory allocated : res  " << v_res_gpu.size()*sizeof(float)/1048576. << " [mB]\n ";
    std::cout << " sum cpu " << sum_cpu << " sum gpu original " << sum_gpu_original << " sum gpu tune " << sum_gpu_tune << std::endl;
    for(int i = 0 ; i < size-1; ++i)
       std::cout << " reduction: "<< i << " range ["  << v_offset[i] <<","<< v_offset[i+1]  << "], cpu:" << v_res[i]  << ", gpu:" << v_res_gpu[i] << ", gpu2:" << v_res_gpu2[i]<< std::endl;

    // insert code here...
    std::cout << " time cpu:" << elapsed_seconds.count()*1000 << " [ms], gpu original " << milliseconds_original << " [ms], gpu tune " << milliseconds_64  << " [ms] \n ";
    return 0;
}
