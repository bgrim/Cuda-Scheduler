#include <stdio.h>

__global__ void clock_block(int kernel_time, int clockRate, int *d_result)
{ 
    int finish_clock;
    int start_time;
    for(int temp=0; temp<kernel_time; temp++){
        start_time = clock();
        finish_clock = start_time + clockRate;
        bool wrapped = finish_clock < start_time;
        while( clock() < finish_clock || wrapped) wrapped = clock()>0 && wrapped;
    }
    (*d_result)= kernel_time;
}

void sleep_setup(cudaStream_t stream, char *filename, void *setupResult)
{
    //open file
    FILE * ftp;
    ftp = fopen(filename,"r");

    // read the kernel_time from the file
    int kernel_time;
    fscanf(ftp, "%d", &kernel_time);

    // set the value of setupResult to the address of kernel_time
    setupResult = (void *) &kernel_time;

    fclose(ftp);
}

void sleep(cudaStream_t stream, void *setupResult)
{
    int *kernel_time = (int *) setupResult
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);	
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    int clockRate = deviceProp.clockRate;

    clock_block<<<1,1,1,stream>>>(*kernel_time, clockRate, d_result);
}

void sleep_finish(char *filename, void *setupResult)
{
    int *kernel_time = (int *)setupResult;
    FILE *out=fopen("matrixOut.txt", "w");

    fprintf(out, "Finished sleep of duration: %d", *kernel_time);

    fclose(out);
}







