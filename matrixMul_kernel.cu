//declar globals and structs
struct timeval tp;
double getTime_sec();
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);
bool check(float*, float*, int, float);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

//    END OF KERNEL                                                                                                                                                                                        
// start CUDA mat mul

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
 // Allocates a matrix with random float entries.                                                                                            

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
      data[i] = rand() / (float)RAND_MAX;
}

void matrixMul(cudaStream_t s, int side_length, int* d_result){

  int cuda_device;
  cudaDeviceProp deviceProp;

  cudaGetDevice(&cuda_device);
  cudaGetDeviceProperties(&deviceProp, cuda_device);

  // use a larger block size for Fermi and above                                                                                                                                                          
  int block_size = (deviceProp.major < 2) ? 16 : 32;

  //  printf("Device %d: \"%s\" with Compute %d.%d capability\n\n", cuda_device, deviceProp.name, deviceProp.major, deviceProp.minor);

  // set seed for rand()                                                                                                                                                                              
  srand(2006);

  //Get command line arguements                                                                                                                                                                           
  int nIter = 30;
  int size = 640;

  unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
  uiWA = size;
  uiHA = size;
  uiWB = size;
  uiHB = size;
  uiWC = size;
  uiHC = size;

  // allocate host memory for matrices A and B                                                                                                                                                            
  unsigned int size_A = uiWA * uiHA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*)malloc(mem_size_A);
  unsigned int size_B = uiWB * uiHB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*)malloc(mem_size_B);

  unsigned int size_C = uiWC * uiHC;
  unsigned int mem_size_C = sizeof(float) * size_C;

  // initialize host memory                                                                                                                                                                               
  randomInit(h_A, size_A);
  randomInit(h_B, size_B);

  // allocate host memory for the result                                                                                                                                                                  
  float* h_C      = (float*) malloc(mem_size_C);

  // allocate device memory                                                                                                                                                                               
  float* d_A, *d_B, *d_C;

  cudaMalloc((void**) &d_A, mem_size_A);
  cudaMalloc((void**) &d_B, mem_size_B);
  cudaMalloc((void**) &d_C, mem_size_C);

  // copy host memory to device                                                                                                                                                                           
  cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

  // setup execution parameters                                                                                                                                                                           
  dim3 threads(block_size, block_size);
  dim3 grid(uiWC / threads.x, uiHC / threads.y);

  //Print information about test                                                                                                                                                                          
  printf("Calculating: C = A x B, %d times\n", nIter);
  printf("Matrix A is :  %d x %d\n", uiWA, uiHA);
  printf("Matrix B is :  %d x %d\n", uiWB, uiHB);
  printf("Matrix C is :  %d x %d\n\n", uiWC, uiHC);

  // call the cudaMatrixMul cuda function
    cudaMatrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, side_length, side_length);
    //int hard_coded = 100;
    //cudaMatrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, hard_coded, hard_coded);
}
