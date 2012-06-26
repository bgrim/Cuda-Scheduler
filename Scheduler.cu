//do some include stuff
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "matrixMul_kernel.cu"
#include "sleep_kernel.cu"
#include "queue.c"
#include <pthread.h>

// set the default value of the kernel time to 1 second


/////////////////////////////////////////////////////////////////
// Global Variables
/////////////////////////////////////////////////////////////////

int kernel_time = 1000;

double startTime_ms;
struct timeval tp;

Queue Q;
pthread_mutex_t queueLock;


////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////

double getTime_msec() {
   gettimeofday(&tp, NULL);
   return static_cast<double>(tp.tv_sec) * 1E3
           + static_cast<double>(tp.tv_usec) / 1E3;
}

cudaStream_t getStream()
{
    bool waiting = true;
    cudaStream_t stream;
    while(waiting)
    {
        pthread_mutex_lock(&queueLock);
        waiting = IsEmpty(Q);
        if(!waiting)
        {
            stream = Front(Q);
            Dequeue(Q);
        }
        pthread_mutex_unlock(&queueLock);
        if(waiting) pthread_yield();
    }
    return stream;
}

void putStream(cudaStream_t stream){
    pthread_mutex_lock(&queueLock);
    Enqueue(stream, Q);
    pthread_mutex_unlock(&queueLock);
}

void *waitOnStream( void *str )
{
    cudaStream_t stream = (cudaStream_t) str;
    cudaError_t cuda_error = cudaStreamSynchronize(stream);
    
    putStream(stream);

    if(cuda_error==cudaSuccess){
        printf( "A thread has finished in  %.4f ms\n", getTime_msec() - startTime_ms);
    }else{
        printf("CUDA Error: %s\n", cudaGetErrorString(cuda_error));
    }
    return 0;
}

char* getNextKernel()
{
    return "sleep";
}

void call(char *kernel)
{
    if(kernel=="sleep")
    {
        cudaStream_t stream = getStream();
        sleep(stream, kernel_time);

        pthread_t thread1;
        int rc = pthread_create( &thread1, NULL, waitOnStream, (void *) stream);
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

    pthread_mutex_init(&queueLock, NULL);

    int throttle = 16;  //this should be set using device properties

    int jobs = 64;

    Q = CreateQueue(throttle);

    if( argc>3 ){
        throttle = atoi(argv[1]);
        jobs = atoi(argv[2]);
        kernel_time = atoi(argv[3]);
    }

    // allocate and initialize an array of stream handles
    //cudaStream_t *streams = (cudaStream_t*) malloc(throttle * sizeof(cudaStream_t));
    
    for(int i = 0; i < throttle; i++)
    {
        cudaStream_t s;
        cudaStreamCreate(&s);
        Enqueue(s, Q);
    }

    char *kernel = "none";

    printf("starting\n");

    for(int k = 0; k<jobs; k++) //later will probably just be true.
    {
        while( kernel == "none" ){
            kernel = getNextKernel();
        }
        call(kernel);
        kernel = "none";
    }
    
    // print out some default information                                                                                                                                                                  
    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    int est = (jobs/throttle)*kernel_time;
    printf("The estimated time should be: %d\n",est);

    cudaError cuda_error = cudaDeviceSynchronize();
    if(cuda_error==cudaSuccess){
        //printf( "Final: Running the Scheduler was a success\n");
    }else{
        printf("Final: CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    // release resources
    DisposeQueue(Q);

    pthread_mutex_destroy(&queueLock);
    return 0;    
}






