#include <stdio.h>
#include <stdlib.h>

//declare globals and structs and headers
struct timeval tp;
double getTime_sec();
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);
bool check(float*, float*, int, float);
void matMul_setup(cudaStream_t s, char *filename, void *matrixSetupResults);
void specificInit(float* data, int side_length);
void writeMatrixToFile(char *filename, float* h_result, int side_length);
int getMatrixSideLengthFromFile(char* filename);
//void matMul_finish(char *filename, void *setupResults);

struct matMulRecord{
  float* h_arrayA;
  float* d_arrayA;
  float* d_results;
  int side_length;
};

//////////////////////////////////////////////////////////////////////////
// Utilities
//////////////////////////////////////////////////////////////////////////

int getMatrixSideLengthFromFile(char* filename){
  int size=0;
  FILE* ftp;
  ftp=fopen(filename, "r");
  fscanf(ftp, "%d", &size);
  return size;
}

void randomInit(float* data, int side_length)
{
  for (int i = 0; i < side_length*side_length; ++i)
    data[i] = rand() / (float)RAND_MAX;
}


////////////////////////////////////////////////////////////////////////////////////
// The kernel
////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE> __global__ void
cudaMatrixMul( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}



////////////////////////////////////////////////////////////////////////
// Functions called by Scheduler to setup the kernel
////////////////////////////////////////////////////////////////////////

// matMul_Setup
void *matMul_setup(cudaStream_t s, char* filename){

  // get side length of file
  // faster for get lines in file or from single line
  int side_length = getMatrixSideLengthFromFile(filename);

  // host array A (the matrix to be squared)                                                                                                                                                            
  float* h_arrayA = (float*)malloc(side_length*side_length*sizeof(float));

  // device array A                                                                                                                                                                                   
  float* d_arrayA;
  cudaMalloc(&d_arrayA, side_length*side_length*sizeof(float));

  // device results                                                                                                                                                                                   
  float* d_results;
  cudaMalloc(&d_results, side_length*side_length*sizeof(float));

  // build arrays                                                                                                                                                                                     
  randomInit(h_arrayA, side_length);

  printf("side length: %d\n", side_length);

  // move to device                                                                                                                                                                                   
  cudaMemcpyAsync(d_arrayA, h_arrayA, side_length*side_length*sizeof(float), cudaMemcpyHostToDevice, s);

  // get a record out of the matrixSetupResults
  matMulRecord *r = (matMulRecord *) malloc(sizeof(struct matMulRecord));
  r->h_arrayA = h_arrayA;
  r->d_arrayA = d_arrayA;
  r->d_results = d_results;
  r->side_length = side_length;

  return (void *) r;
}

// CPU wrapper method matrixMul for launching CUDA kernel
void matrixMul(cudaStream_t s, void *setupResults){
  int block_size = 32;

  matMulRecord *r = (matMulRecord *) setupResults;

  float *d_arrayA = r->d_arrayA;
  float *d_results = r->d_results;
  int side_length = r->side_length;

  // setup execution parameters                                                           
  dim3 threads(block_size, block_size);
  dim3 grid(side_length / threads.x, side_length / threads.y);

  // call the cudaMatrixMul cuda function
  cudaMatrixMul<32><<< grid, threads, 0, s>>>(d_results, d_arrayA, d_arrayA, side_length, side_length);
}

void matMul_finish(cudaStream_t s, char *filename, void *setupResults){
  matMulRecord *r = (matMulRecord *) setupResults;

  float *h_arrayA = r->h_arrayA;
  float *d_arrayA = r->d_arrayA;
  float *d_results = r->d_results;
  int side_length = r->side_length;

  // This allocates an intermediate host array, h_results,
  float* h_results = (float*)malloc(side_length*side_length*sizeof(float));

  // then copies results from d_results to h_results
  cudaMemcpyAsync(h_results, d_results, side_length*side_length*sizeof(float), cudaMemcpyDeviceToHost, s);

  // then deallocates all arrays
  cudaFree(d_results);
  cudaFree(d_arrayA);
  free(h_arrayA);
  free(h_results);
  free(r);
}










