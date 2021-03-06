#include <stdio.h>

//__global__ void clock_block(int kernel_time, int clockRate, int *d_result)
__global__ void clock_block(int kernel_time, int clockRate)
{ 
    int finish_clock;
    int start_time;
    for(int temp=0; temp<kernel_time; temp++){
        start_time = clock();
        finish_clock = start_time + clockRate;
        bool wrapped = finish_clock < start_time;
        while( clock() < finish_clock || wrapped) wrapped = clock()>0 && wrapped;
    }
    //    (*d_result)= kernel_tim;e

}

//void sleep(cudaStream_t stream, int kernel_time, int *d_result){
void sleep(cudaStream_t stream, int kernel_time){


    int cuda_device = 7;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);	
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    int clockRate = deviceProp.clockRate;

    //    clock_block<<<1,1,1,stream>>>(kernel_time, clockRate, d_result);
    clock_block<<<1,1,1,stream>>>(kernel_time, clockRate);

}
