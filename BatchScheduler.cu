//do some include stuff
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "matrixMul_kernel.cu"
#include "sleep_kernel.cu"

// set the default value of the kernel time to 1 second


/////////////////////////////////////////////////////////////////
// Global Variables
/////////////////////////////////////////////////////////////////

int kernel_time = 1000;

struct timeval tp;

double startTime_ms;

bool randomSleeps;

////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////

double getTime_msec() {
   gettimeofday(&tp, NULL);
   return static_cast<double>(tp.tv_sec) * 1E3
           + static_cast<double>(tp.tv_usec) / 1E3;
}

char* getNextKernel()
{
    return "sleep";
}
int getKernelParam()
{
    if (randomSleeps)
        return (int)(kernel_time*(rand()/(RAND_MAX+1.0)));
    else
        return kernel_time;
}


void call(char *kernel, cudaStream_t stream, int param)
{
    if(kernel=="sleep")
    {
        sleep(stream, param);
    }
}

void printAnyErrors()
{
    cudaError e = cudaGetLastError();
    if(e!=cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString( e ) );
    }
}



////////////////////////////////////////////////////////////////////
// The Main
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    startTime_ms = getTime_msec();

    int throttle = 16;  //this should be set using device properties

    int jobs = 64;

    randomSleeps = false;

    if( argc>3 ){
        throttle = atoi(argv[1]);
        jobs = atoi(argv[2]);
        kernel_time = atoi(argv[3]);
    }
    if( argc>4 )
    {
        randomSleeps = true;
        srand(atoi(argv[4]));
    }

    cudaStream_t *streams = (cudaStream_t *) malloc(throttle*sizeof(cudaStream_t));

    for(int i = 0; i < throttle; i++)
    {
      cudaStreamCreate(&streams[i]);
    }

    char *kernel = "none";

    printf("starting\n");

    int totalTime = 0;

    for(int k = 0; k<jobs;) //later will probably just be true.
    {
        for(int j=0;j<throttle && k<jobs;j++){

            int param;

            while( kernel == "none" ){
                kernel = getNextKernel();
                param = getKernelParam();
            }
            totalTime+=param;
            call(kernel, streams[j], param);
            k++;

            kernel = "none";
        }
        cudaDeviceSynchronize();
    }

    // release resources

    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    printf("The total time slept is: %d\n", totalTime);
    if(!randomSleeps)
        printf("The estimated time is: %d\n\n", (((jobs-1)/throttle)+1)*kernel_time);


    for(int i =0; i<throttle; i++) cudaStreamDestroy(streams[i]);

    free(streams);

    return 0;    
}






