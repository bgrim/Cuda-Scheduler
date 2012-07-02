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

// commenting out the tp because it is in the include
//struct timeval tp;

double startTime_ms;

////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////

double getTime_msec() {
   gettimeofday(&tp, NULL);
   return static_cast<double>(tp.tv_sec) * 1E3
           + static_cast<double>(tp.tv_usec) / 1E3;
}

int getNextKernel()
{
  // for now always select mat mul
    return 2;
}

int getKernelParam()
{
    return side_length;
}

void call(int kernel, cudaStream_t stream, float* param, float* d_result)
{
  printf("I'm in call");
    // sleep
  //    if(kernel==1)
  // {
  //    sleep(stream, param, d_result);
  //}
    // mat mul
    if(kernel==2)
    {
      matrixMul(stream, side_length, param, d_result);
    }
}

void printAnyErrors()
{
    cudaError e = cudaGetLastError();
    if(e!=cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString( e ) );
    }
}

// Allocates a matrix with random float entries.                                                            
void randomInit(float* data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}
// Allocates a matrix with random float entries.                                                            
void specificInit(float* data, int totalSize)
{
  // open a file
  // read contents into a buffer
  for (int i = 0; i < totalSize; ++i)
    // read the next int from the buffer

}

////////////////////////////////////////////////////////////////////
// The Main
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    startTime_ms = getTime_msec();

    int throttle = 16;  //this could be set using device properties

    int jobs = 64;

    if( argc>3 ){
        throttle = atoi(argv[1]);
        jobs = atoi(argv[2]);
	side_length = atoi(argv[3]);
    }
    
    printf("The number of jobs is equal to: %d\n", jobs);
    printf("The side_length is equal to: %d\n", side_length);

    cudaStream_t *streams = (cudaStream_t *) malloc(throttle*sizeof(cudaStream_t));

    printf("create streams\n");
    for(int i = 0; i < throttle; i++)
    {
      cudaStreamCreate(&streams[i]);
    }

    int kernel = 0;

    printf("starting\n");

    // loop for number of batches
    for(int k = 0; k<jobs;) //later will probably just be true.
    {
      printf("made it into for loop\n");

        int jobsLaunched = 0; 

	printf("make kernels\n");
	// array for kernel
	int *kernels = (int *) malloc(throttle*sizeof(int));
	printf("make params\n");
	// array for parameter
	int *parameters = (int *) malloc(throttle*sizeof(int));

	printf("reading jobs\n");
	for(int q=0; q<throttle; q++){
	  kernels[q] = getNextKernel();
	  parameters[q] = getKernelParam();
	}

	// hard code all side lengths to be the same
	side_length = parameters[0];

	// host array A
	float* h_arrayA = (float*)malloc(throttle*side_length*side_length*sizeof(float));

	// device array A
	float* d_arrayA;
	cudaMalloc(&d_arrayA, throttle*side_length*side_length*sizeof(float));

	// array C
	float* h_results = (float*)malloc(throttle*side_length*side_length*sizeof(float));

	// malloc on dev
        float* d_results;
        cudaMalloc(&d_results, throttle*side_length*side_length*sizeof(float));     

	printf("build arrays\n");
	// build arrays
	for(int p=0; p<throttle; p++){
	  // allocate matrix A
	  printf("In loop: %d\n", p);
	  //	  randomInit(&h_arrayA[p], side_length*side_length);
	  specificInit(&h_arrayA[p], side_length*side_length);
	}

	printf("starting copy to device.\n");
	// move to device
	cudaMemcpy(d_arrayA, h_arrayA, throttle*side_length*side_length*sizeof(float), cudaMemcpyHostToDevice);

	// loop over each batch
        for(int j=0;j<throttle && k<jobs;j++){
	  printf("Launching batch %d\n", j);
            float* param = &d_arrayA[j];
     
            call(kernel, streams[j], param, &d_results[j]);

            jobsLaunched++;
            k++;

            kernel = 0;
        }

        cudaError err = cudaDeviceSynchronize();
        printf("finished a batch: %s\n", cudaGetErrorString( err ) );

	// write to host
	cudaMemcpy(h_results, d_results, throttle*side_length*side_length*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_results);
	cudaFree(d_arrayA);
	free(kernels);
	free(parameters);
	free(h_arrayA);
        free(h_results);
    }

    // release resources

    printf("The number of jobs equals: %d\n",jobs);
    printf("The current throttle is: %d\n", throttle);
    // printf("The estimated time is: %d\n\n", (((jobs-1)/throttle)+1)*kernel_time);

    for(int i =0; i<throttle; i++) cudaStreamDestroy(streams[i]);

    free(streams);

    return 0;    
}
