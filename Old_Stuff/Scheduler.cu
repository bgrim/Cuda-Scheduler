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

double startTime_ms;

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

record* getStream()
{
    bool waiting = true;
    record *r;
    //cudaStream_t stream;
    while(waiting)
    {
        pthread_mutex_lock(&queueLock);
        waiting = IsEmpty(Q);
        if(!waiting)
        {
	  //stream = Front(Q);
	    r = Front(Q);
            Dequeue(Q);
        }
        pthread_mutex_unlock(&queueLock);
        if(waiting) pthread_yield();
    }
    return r;
}

void putStream(record *r){
    //bool waiting = true;
    //while(waiting)
    //{
        pthread_mutex_lock(&queueLock);
        //waiting = IsFull(Q);
        //if(!waiting) Enqueue(stream, Q);
        Enqueue(r, Q);    //extra line
        pthread_mutex_unlock(&queueLock);
        //if(waiting) pthread_yield();
    //}
}

void *waitOnStream( void *arg )
{
  //    cudaStream_t stream = (cudaStream_t) arg;
    record *r = (record *) arg;

    // cudaEvent_t event;
    // cudaEventCreate(&event); 
    // cudaEventRecord(event, stream);
    // cudaEventSynchronize(event);
    // cudaEventDestroy(event);
    
    //cudaStreamSynchronize(r->stream);

    double time = getTime_msec();
    while(cudaSuccess!=cudaStreamQuery(r->stream)){
        //while(getTime_msec()<time+500);
    }

    printf(" done waiting for kernel at %.4f ms in stream: %d\n",getTime_msec() - startTime_ms, r->index);

    putStream(r);

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
        record *r = getStream();
        printf("   main at time %.2f ms in stream: %d\n", getTime_msec()-startTime_ms, r->index);
        pthread_t manager;
        sleep(r->stream, kernel_time);
        pthread_create( &manager, NULL, waitOnStream, (void *) r);
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

    cudaStream_t *streams = (cudaStream_t *) malloc(throttle*sizeof(cudaStream_t));

    // create record array
    record **recordArray = (record **) malloc(throttle*sizeof(record *));


    for(int i = 0; i < throttle; i++)
    {
      // create a new record with the cuda stream create and the loop counter as the index
      cudaStreamCreate(&streams[i]);
      // allocate the record
      record *r = (record *) malloc (sizeof(struct record));
      r->stream = streams[i];
      r->index = i;
      Enqueue(r, Q);
      recordArray[i] = r;
    }

    char *kernel = "none";

    printf("starting\n");

    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    int est = (jobs/throttle)*kernel_time;
    printf("The estimated time should be: %d\n\n",est);

    for(int k = 0; k<jobs; k++) //later will probably just be true.
    {
        //while( kernel == "none" ){
        //    kernel = getNextKernel();
        //}
        kernel = "sleep";
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

    // free each element of the array
    for(int i = 0; i<throttle; i++)
    {
        free(recordArray[i]);
    }
    free(recordArray);

    free(streams);
    DisposeQueue(Q);
    
    for(int i=0; i<100000; i++);
    
    pthread_mutex_destroy(&queueLock);
    return 0;    
}






