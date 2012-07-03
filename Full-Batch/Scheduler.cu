#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "matrixMul_kernel.cu"
#include "sleep_kernel.cu"


/////////////////////////////////////////////////////////////////
// Global Variables
/////////////////////////////////////////////////////////////////

double startTime_ms;  //this is helpful for debugging sometimes
                      //its vlue is the first thing set by the program

////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////
double getTime_msec() {
   gettimeofday(&tp, NULL);
   return static_cast<double>(tp.tv_sec) * 1E3
           + static_cast<double>(tp.tv_usec) / 1E3;
}

//Eventually getNextKernel(), getKernelInput() and getKernelOutput() will interact
//  with a daemon that is listening for job information.
int getNextKernel()
{
    return 2;
}
char *getKernelInput()
{
    return "matrixIn.txt";
}
char *getKernelOutput(){
    return "matrixOut.txt"; 
}



//This method will let whatever kernel is about to run setup any device memory it needs
//  and do any file I/O needed. All Asynchronous operations will be in stream
void *kernel_setup(int kernel, cudaStream_t stream, char * filename)
{
    if(kernel==1) return sleep_setup(stream, filename);

    if(kernel==2) return matMul_setup(stream, filename);

    return (void *) 1;
}

//This method will launch the given kernel in stream with setupResults.
void kernel_call(int kernel, cudaStream_t stream, void *setupResults)
{
    if(kernel==1) sleep(stream, setupResults);

    if(kernel==2) matrixMul(stream, setupResults);
}

//This method will let the kernel deallocate all the memory that it acquired in
//  kernel_setup and also lets the kernel write to its output file.
void kernel_finish(int kernel, char * filename, void *setupResult )
{
    if(kernel==1) sleep_finish(filename, setupResult);

    if(kernel==2) matMul_finish(filename, setupResult);
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
    if( argc>3 ){
        throttle = atoi(argv[1]);
        jobs = atoi(argv[2]);
    }
   
    printf("The number of jobs is equal to: %d\n", jobs);

    //make throttle many streams to run concurrent kernels in
    cudaStream_t *streams = (cudaStream_t *) malloc(throttle*sizeof(cudaStream_t));
    for(int i = 0; i < throttle; i++)
	cudaStreamCreate(&streams[i]);


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
	    kernels[q] = getNextKernel();
	    inputFiles[q] = getKernelInput();
            outputFiles[q] = getKernelOutput();
            k++;
            batchSize++;
	}

	// An array containing the state that each kernel needs
	void **setupResults = (void **) malloc(throttle*sizeof(void *));

	// Let each kernel read its input file and fill its setupResult
	for(int q=0; q<batchSize; q++){
	    setupResults[q] = kernel_setup(kernels[q], streams[q], inputFiles[q]);
            printf("Ugh: %d\n", ((matMulRecord *) setupResults[q])->side_length);
	}

	// call each kernel in a different stream giving it its setupResult
        for(int q=0; q<batchSize; q++){
            kernel_call(kernels[q], streams[q], setupResults[q]);
        }

	// wait for all kernels to finish
        cudaError err = cudaDeviceSynchronize();
	printf("finished a batch: %s\n", cudaGetErrorString( err ) );

	// let each kernel copy its results back and write to its output file
	// they should do there own clean up (i.e. memory deallocate and closing files)
	for(int q=0; q<batchSize; q++){
	    kernel_finish(kernels[q], outputFiles[q], setupResults[q]);
	}
	printf("finished kernel_finish");
	//free the arrays that we used;
	free(kernels);
	free(inputFiles);
	free(outputFiles);
    }

    cudaError err = cudaDeviceSynchronize();
    printf("finished all jobs: %s\n", cudaGetErrorString( err ) );
    // release resources

    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    // printf("The estimated time is: %d\n\n", (((jobs-1)/throttle)+1)*kernel_time);

    for(int i =0; i<throttle; i++) cudaStreamDestroy(streams[i]);

    free(streams);

    return 0;    
}







