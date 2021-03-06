//do some include stuff
#include <stdio.h>
#include <cuda_runtime.h>
#include "matrixMul_kernel.cu"
#include "sleep_kernel.cu"

int kernel_time = 1000;

////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////
char* getNextKernel()
{
    return "sleep";
}

void call(char *kernel, cudaStream_t stream)
{
    if(kernel=="sleep")  //This will eventually take better parameters and
                         // have more different kernels to call
    {
        int cuda_device = 0;
        cudaDeviceProp deviceProp;
        cudaGetDevice(&cuda_device);	
        cudaGetDeviceProperties(&deviceProp, cuda_device);
        int clockRate = deviceProp.clockRate;

        clock_block<<<1,1,1,stream>>>(kernel_time, clockRate);  
        //currently hard coded time
    }
}

void printAnyErrors(){
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
    //Default values
    int throttle = 16;  //this should be set using device properties

    int cuda_device = 0; //Default, a better version would utilize every cuda
                         // enabled device and schedule across all of them

    // read in the arg for the kernel time if available
    if( argc>1 ){
        kernel_time = atoi(argv[1]);
    }

    //Getting device information, because we need clock_rate later
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);	
    cudaGetDeviceProperties(&deviceProp, cuda_device);

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(throttle * sizeof(cudaStream_t));
    for(int i = 0; i < throttle; i++)
        cudaStreamCreate(&(streams[i]));

    // set the default kernel to be equal to none
    char *kernel = "none";
    int i = 0;

    printf("starting\n");

    // number of jobs to run
    int jobs = 64;

    // loop through all of the jobs
    for(int k = 0; k<jobs; k++) //later will probably just be true.
    {
        while( kernel == "none" ){
            kernel = getNextKernel();
        }
        call(kernel, streams[i]);

        i = (i+1)%throttle;  //This is a bad way to do this. We should keep track
                             // of which streams are currently not executing anything
                             // which will require us to make a cpu thread wait on each
                             // kernel. We can do passive waits with events

        printAnyErrors(); //This should also be called by a cpu thread waiting for each
                          // kernel to finish
        kernel = "none";
    }
    

    // print out some default information
    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    int est = (jobs/throttle)*kernel_time;
    printf("The estimated time should be: %d",est);

    cudaError cuda_error = cudaDeviceSynchronize();
    if(cuda_error==cudaSuccess){
        printf( "  Running the Scheduler was a success\n");
    }else{
        printf("CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    printAnyErrors(); 
    // release resources
    for(int i = 1; i < throttle; i++)
        cudaStreamDestroy(streams[i]); 
 
    free(streams);
  return 0;    
}






