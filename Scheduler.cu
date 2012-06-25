//do some include stuff
#include <stdio.h>
#include <cuda_runtime.h>
#include "matrixMul_kernel.cu"
#include "sleep_kernel.cu"
//#include "queue.c"
#include <pthread.h>

// set the default value of the kernel time to 1 second


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

  // original code
    if(kernel=="sleep")  //This will eventually take better parameters and
                         // have more different kernels to call
    {
        pthread_t thread1;
        int rc = pthread_create( &thread1, NULL, sleep, (void *) stream);
        if(rc){
            printf("pthread had error and returned %d\n", rc);
        }
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
    int throttle = 8;  //this should be set using device properties

    int cuda_device = 0; //Default, a better version would utilize every cuda
                         // enabled device and schedule across all of them

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

    char *kernel = "none";
    int i = 0;

    printf("starting\n");

    for(int k = 0; k<16; k++) //later will probably just be true.
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
    
    //I should never leave that loop!!
    cudaError cuda_error = cudaDeviceSynchronize();
    if(cuda_error==cudaSuccess){
        printf( "Final: Running the Scheduler was a success\n");
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






