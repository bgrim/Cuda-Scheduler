#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "matrixMul_kernel.cu"
#include "sleep_kernel.cu"
#include "matrixMul_cpu.c"
#include "sleep_cpu.c"

#include "daemon.c"
#include "cleaner.cu"
#include <pthread.h>

/////////////////////////////////////////////////////////////////
// Global Variables
/////////////////////////////////////////////////////////////////

double startTime_ms;  //this is helpful for debugging sometimes
                      //its vlue is the first thing set by the program
struct tp;
////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////
double getTime_msec() {
   gettimeofday(&tp, NULL);
   return static_cast<double>(tp.tv_sec) * 1E3
           + static_cast<double>(tp.tv_usec) / 1E3;
}

//This method will let whatever kernel is about to run setup any device memory it needs
//  and do any file I/O needed. All Asynchronous operations will be in stream
void *kernel_setup(int kernel, cudaStream_t stream, char * filename)
{
    if(kernel==1) return sleep_setup(stream, filename);

    if(kernel==2) return matMul_setup(stream, filename);
    
    if(kernel==3) return sleep_cpu(filename);

    if(kernel==4) return matMul_cpu(filename);

    return (void *) 1;
}

//This method will launch the given kernel in stream with setupResults.
void kernel_call(int kernel, cudaStream_t stream, void *setupResults)
{
    if(kernel==1) sleep(stream, setupResults);

    if(kernel==2) matrixMul(stream, setupResults);
}


//prints the most recent error that hasn't been printed before
//does nothing if there are no errors
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

    //sets the throttle and number of jobs based on inputs or defaults
    int throttle = 16;
    int jobs = 64;
    if( argc>2 ){
        throttle = atoi(argv[1]);
        jobs = atoi(argv[2]);
    }

    daemon_init(jobs);
    pthread_t daemon;
    pthread_create(&daemon, NULL, daemon_Main, (void *) &jobs );

    pthread_mutex_t cleanerLock;
    pthread_mutex_init(&cleanerLock, NULL);

    printf("The number of jobs is equal to: %d\n", jobs);

    //make throttle many streams to run concurrent kernels in

    int batchNum = 0;

    // loop for number of batches
    for(int k = 0; k<jobs;)
    {
        int batchSize = 0; //this will be throttle or less if we run out of jobs

	// arrays for kernel type and its input/output files
	int *kernels = (int *) malloc(throttle*sizeof(int));
	char **inputFiles = (char **) malloc(throttle*sizeof(char *));
	char **outputFiles = (char **) malloc(throttle*sizeof(char *));

        // get information for throttle many jobs or until we are out of jobs
	for(int q=0; q<throttle && k<jobs; q++){
	    kernels[q] = daemon_GetNextKernel();
	    inputFiles[q] = daemon_GetInputFile();
            outputFiles[q] = daemon_GetOutputFile();

	    printf("Kernel information for kernel  %d  of batch number  %d\n", q, batchNum);
	    printf("kernel: %d\n", kernels[q]);
	    printf("input:  %s\n", inputFiles[q]);
	    printf("output: %s\n\n", outputFiles[q]);

            k++;
            batchSize++;
	}
        //allocate a stream for each kernel
        cudaStream_t *streams = (cudaStream_t *) malloc(batchSize*sizeof(cudaStream_t));
        for(int i = 0; i < batchSize; i++)
	    cudaStreamCreate(&streams[i]);

	// An array containing the state that each kernel needs
	void **setupResults = (void **) malloc(throttle*sizeof(void *));

	// Let each kernel read its input file and fill its setupResult
	for(int q=0; q<batchSize; q++){
	    setupResults[q] = kernel_setup(kernels[q], streams[q], inputFiles[q]);
	}

        pthread_mutex_lock(&cleanerLock);

	// call each kernel in a different stream giving it its setupResult
        for(int q=0; q<batchSize; q++){
            kernel_call(kernels[q], streams[q], setupResults[q]);
        }

        pthread_mutex_unlock(&cleanerLock);

	// wait for all kernels to finish
        cudaError e = cudaDeviceSynchronize();
	printf("finished batch number: %d  with error: %s\n\n",batchNum,cudaGetErrorString(e));

        pthread_t cleaner;

        CleanerRecord *params = makeCleanerRecord(batchSize, streams, inputFiles, 
                                                    outputFiles, kernels, setupResults, cleanerLock);

        pthread_create(&cleaner, NULL, cleaner_Main, (void *) params );

        batchNum++;
    }

    cudaError err = cudaDeviceSynchronize();
    printf("finished all jobs and cleaner with error: %s\n\n", cudaGetErrorString( err ) );
    // release resources

    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);

    daemon_free();

    pthread_mutex_destroy(&cleanerLock);

    return 0;    
}







