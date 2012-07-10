
//////////////////////////////////////////////////////////////////////////
// Utilities
//////////////////////////////////////////////////////////////////////////

int getMatrixSideLengthFromFile_cpu(char* filename){
  int size=0;
  FILE* ftp;
  ftp=fopen(filename, "r");
  fscanf(ftp, "%d", &size);
  return size;
}

void randomInit_cpu(float* data, int side_length)
{
  for (int i = 0; i < side_length*side_length; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void multiply(int side, float *in, float *out)
{
  int i, j, k;
  for(i=0; i<side;i++){
    for(j=0; j<side;j++){
      float sum = 0;
      for(k=0; k<side; k++){
	sum+=(in[k+j*side]*in[i+k*side]);  
      }
      out[i+j*side]=sum;
    }
  }
}


////////////////////////////////////////////////////////////////////////
// Functions called by Scheduler to setup the kernel
////////////////////////////////////////////////////////////////////////

// matMul_Setup
void *matMul_cpu(char* filename){

  // get side length of file
  // faster for get lines in file or from single line
  int side_length = getMatrixSideLengthFromFile_cpu(filename);

  // host array A (the matrix to be squared)                                                                                                                                                            
  float* h_arrayA = (float*)malloc(side_length*side_length*sizeof(float));

  float *h_result = (float *)malloc(side_length*side_length*sizeof(float));

  randomInit_cpu(h_arrayA, side_length);

  printf("side length: %d\n", side_length);

  multiply(side_length, h_arrayA, h_result);

  return (void *) 1;
}









