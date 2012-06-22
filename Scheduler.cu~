

//do some include stuff
#include <stdio.h>
#include <cuda_runtime.h>
#include <matrixMul_kernel.cu>
#include <sleep_kernel.cu>

////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////

void call(char *kernel, cudaStream_t stream)
{
    if(kernel=="sleep")  //This will eventually take better parameters and
                         // have more different kernels to call
    {
        cudaDeviceProp deviceProp;
        cudaGetDevice(&cuda_device);	
        cudaGetDeviceProperties(&deviceProp, cuda_device);
        int clockRate = deviceProp.clockRate;

        clock_block<<<1,1,1,stream>>>(1000, clockRate);  
                             //currently hard coded to run for 1000ms
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

    if( argc>2 ){
        throttle = atoi(argv[1]);
    }

    //Getting device information, because we need clock_rate later
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);	
    cudaGetDeviceProperties(&deviceProp, cuda_device);


    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(throttle * sizeof(cudaStream_t));
    for(int i = 0; i < throttle; i++)
        cudaStreamCreate(&(streams[i]));

    char *kernel = "none" 
    int i = 0;
    while(true)
    {
        while( kernel = "none" ){
            kernel = getNextKernel();
        }
        call(kernel, streams[i]);

        i = (i+1)%throttle;  //This is a bad way to do this. We should keep track
                             // of which streams are currently not executing anything
                             // which will require us to make a cpu thread wait on each
                             // kernel. We can do passive waits with events

        printAnyErrors(); //This should also be called by a cpu thread waiting for each
                          // kernel to finish
        kernel = "none"
    }

    //I should never leave that loop!!

    printAnyErrors(); 
    // release resources
    for(int i = 1; i < nstreams; i++)
        cudaStreamDestroy(streams[i]); 
 
    free(streams);
  return 1;    
}






