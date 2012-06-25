//do some include stuff
#include <stdio.h>
#include <cuda_runtime.h>
#include "matrixMul_kernel.cu"
#include "sleep_kernel.cu"
//#include "queue.c"
#include <pthread.h>

// set the default value of the kernel time to 1 second


int kernel_time = 1000;
pthread_mutex_t mutexsum;

////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////

void *waitOnStream( void *str )
{
    cudaStream_t stream = (cudaStream_t) str;
    cudaError_t cuda_error = cudaStreamSynchronize(stream);
    if(cuda_error==cudaSuccess){
        printf( "Running the Scheduler was a success\n");
    }else{
        printf("CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }
    return 0;
}

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
        sleep(stream, kernel_time);

        //pthread_t thread1;
        //int rc = pthread_create( &thread1, NULL, waitOnStream, (void *) stream);
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
    //Default values
    int throttle = 16;  //this should be set using device properties

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

    int jobs = 64;
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
        kernel = "none";
    }
    
    // print out some default information                                                                                                                                                                  
    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    int est = (jobs/throttle)*kernel_time;
    printf("The estimated time should be: %d",est);

    cudaError cuda_error = cudaDeviceSynchronize();
    if(cuda_error==cudaSuccess){
        printf( "Final: Running the Scheduler was a success\n");
    }else{
        printf("CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    // release resources
    for(int i = 1; i < throttle; i++)
        cudaStreamDestroy(streams[i]); 
 
    free(streams);
    pthread_mutex_destroy(&mutexsum);
    return 0;    
}






