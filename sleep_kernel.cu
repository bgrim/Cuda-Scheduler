#include <stdio.h>

__global__ void clock_block(int kernel_time, int clockRate)
{ 
    //This method will sleep for clockRate*kernel_time many clock ticks
    // which is equivalent to sleeping for kernel_time milliseconds
    int finish_clock;
    int start_time;
    for(int temp=0; temp<kernel_time; temp++){
        start_time = clock();
        finish_clock = start_time + clockRate;
        bool wrapped = finish_clock < start_time;
        while( clock() < finish_clock || wrapped) wrapped = clock()>0 && wrapped;
    }
}


void *sleep_setup(cudaStream_t stream, char *filename)
{
    //open file
    FILE * ftp;
    ftp = fopen(filename,"r");

    printf("starting setup with %s\n", filename);

    // read the kernel_time from the file
    int *kernel_time = (int *) malloc(sizeof(int));
    fscanf(ftp, "%d", kernel_time);

    fclose(ftp);

    printf("done with setup of %s\n", filename);

    return (void *) kernel_time;
}

void sleep(cudaStream_t stream, void *setupResult)
{
    //get the kernel time
    int *kernel_time = (int *) setupResult;

    //get the clock rate
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);	
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    int clockRate = deviceProp.clockRate;

    //launch the kernel in the stream
    clock_block<<<1,1,1,stream>>>(*kernel_time, clockRate);
}

void sleep_finish(cudaStream_t s, char *filename, void *setupResult)
{
    // opens the output file
    int *kernel_time = (int *) setupResult;
    FILE *out=fopen(filename, "w");

    //write a nice little messege in it, including kernel_time
    fprintf(out, "Finished sleep of duration: %d", *kernel_time);

    //clean up
    fclose(out);
    free(kernel_time);
}







