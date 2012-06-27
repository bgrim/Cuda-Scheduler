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

struct timeval tp;

struct record{
  cudaStream_t stream;
  int index;
};

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
    //bool waiting = true;
    //while(waiting)
    //{
        pthread_mutex_lock(&queueLock);
        //waiting = IsFull(Q);
        //if(!waiting) Enqueue(stream, Q);
        Enqueue(stream, Q);    //extra line
        pthread_mutex_unlock(&queueLock);
        //if(waiting) pthread_yield();
    //}
}

void *waitOnStream( void *arg )
{
    cudaStream_t stream = (cudaStream_t) arg;

    double startTime_ms = getTime_msec();

    // cudaEvent_t event;
    // cudaEventCreate(&event); 
    // cudaEventRecord(event, stream);
    // cudaEventSynchronize(event);
    // cudaEventDestroy(event);
    
    //    cudaStreamSynchronize(stream);

    //    printf(" done waiting for kernel in %.4f ms\n",getTime_msec() - startTime_ms);

    putStream(stream);

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
        pthread_t manager;
        sleep(stream, kernel_time);
        pthread_create( &manager, NULL, waitOnStream, (void *) stream);
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
    pthread_mutex_init(&queueLock, NULL);

    int throttle = 16;  //this should be set using device properties

    int jobs = 64;

    Q = CreateQueue(throttle);

    if( argc>3 ){
        throttle = atoi(argv[1]);
        jobs = atoi(argv[2]);
        kernel_time = atoi(argv[3]);
    }    

    cudaStream_t *streams = (cudaStream_t *) malloc(throttle*sizeof(cudaStream_t));

    // create record array
    int recordArray[throttle];

    for(int i = 0; i < throttle; i++)
    {
      // create a new record with the cuda stream create and the loop counter as the index
      cudaStreamCreate(&streams[i]);
      // allocate the record
      record r = (record) malloc(throttle*sizeof(struct record));
      r.stream = stream[i];
      r.index = i;
      Enqueue(r, Q);
      // add to record array
    }

    char *kernel = "none";

    printf("starting\n");

    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    int est = (jobs/throttle)*kernel_time;
    printf("The estimated time should be: %d\n\n",est);

    for(int k = 0; k<jobs; k++) //later will probably just be true.
    {
        while( kernel == "none" ){
            kernel = getNextKernel();
        }
        call(kernel);
        kernel = "none";
    }
                                                                                                                                                        
    

    cudaError cuda_error = cudaDeviceSynchronize();
    if(cuda_error==cudaSuccess){
        //printf( "Final: Running the Scheduler was a success\n");
    }else{
        printf("Final: CUDA Error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    // release resources

    for(int i =0; i<throttle; i++) cudaStreamDestroy(streams[i]);

    // loop through record arry
    // free each element of the array

    free(streams);
    DisposeQueue(Q);
    
    for(int i=0; i<100000; i++);
    
    pthread_mutex_destroy(&queueLock);
    return 0;    
}






