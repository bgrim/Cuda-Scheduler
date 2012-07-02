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

//int kernel_time = 1000;

// default side lenth to 100
int side_length = 100;

struct timeval tp;

double startTime_ms;

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
    return "matrix_multiply";
}
void *getKernelParam()
{
    return (void *) &side_length;
}


void call(char *kernel, cudaStream_t stream, void *param, int *d_result)
{
    if(kernel=="sleep")
    {
        sleep(stream, *((int *) param), d_result);
    }
    if(kernel=="matrix_multiply")
    {
        matrixMul(stream, *((int *) param), d_result);
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

    int throttle = 16;  //this could be set using device properties

    int jobs = 64;

    /*
    // get kernel_time if overridden on the command line                                                                                       
    if (cutCheckCmdLineFlag(argc, (const char **)argv, "side_length")) {
      cutGetCmdLineArgumenti(argc, (const char **)argv, "side_length", &side_length);
    }
    // get kernel_time if overridden on the command line                                                                                       
    if (cutCheckCmdLineFlag(argc, (const char **)argv, "jobs")) {
      cutGetCmdLineArgumenti(argc, (const char **)argv, "jobs", &jobs);
    }
    // get kernel_time if overridden on the command line                                                                                       
    if (cutCheckCmdLineFlag(argc, (const char **)argv, "throttle")) {
      cutGetCmdLineArgumenti(argc, (const char **)argv, "throttle", &throttle);
    }
    */

    printf("The side_length is equal to: %d", side_length);

    if( argc>3 ){
        throttle = atoi(argv[1]);
        jobs = atoi(argv[2]);
	// kernel_time = atoi(argv[3]);
	side_length = atoi(argv[3]);
    }

    cudaStream_t *streams = (cudaStream_t *) malloc(throttle*sizeof(cudaStream_t));

    for(int i = 0; i < throttle; i++)
    {
      cudaStreamCreate(&streams[i]);
    }

    char *kernel = "none";

    printf("starting\n");


    for(int k = 0; k<jobs;) //later will probably just be true.
    {
        int *results = (int *) malloc(throttle*sizeof(int));

        int *d_results;
        cudaMalloc(&d_results, throttle*sizeof(int));     

        int jobsLaunched = 0; 
        for(int j=0;j<throttle && k<jobs;j++){

            int *d_result = &d_results[j];
            void *param;

            while( kernel == "none" ){
                kernel = getNextKernel();
                param = getKernelParam();
            }

            call(kernel, streams[j], param, d_result);

            jobsLaunched++;
            k++;

            results[j]=1;
            kernel = "none";
        }

        cudaError err = cudaDeviceSynchronize();
        printf("finished a batch: %s\n", cudaGetErrorString( err ) );

        for(int j=0;j<jobsLaunched;j++){
            //int *d_result = &d_results[j];

            cudaMemcpy(&(results[j]), &(d_results[j]), sizeof(int), cudaMemcpyDeviceToHost);
            printf("A job returned the value: %d\n", results[j]);
        }
        cudaFree(d_results);
        free(results);
    }

    // release resources

    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    // printf("The estimated time is: %d\n\n", (((jobs-1)/throttle)+1)*kernel_time);

    for(int i =0; i<throttle; i++) cudaStreamDestroy(streams[i]);

    free(streams);

    return 0;    
}
