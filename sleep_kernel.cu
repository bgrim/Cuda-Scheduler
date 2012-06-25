#include <stdio.h>

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
}

void *sleep(void *str){
    cudaStream_t stream = (cudaStream_t) str;
    int kernel_time = 1000;

    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);	
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    int clockRate = deviceProp.clockRate;

    clock_block<<<1,1,1,stream>>>(kernel_time, clockRate);

    cudaError_t cuda_error = cudaStreamSynchronize(stream);
    if(cuda_error==cudaSuccess){
        printf( "Running the Scheduler was a success\n");
    }else{
        printf("CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }
    return 0;
}
