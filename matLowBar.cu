#include "matrixMul_kernel.cu"
#include <stdio.h>

//headers

//main
int main(){
  cudaStream_t s;

  char *fileIn = "Inputs/matrixIn0.txt";;
  //  sprintf(fileIn, "Inputs/matrixIn0.txt", i);

  void* setupResults = matMul_setup(s, fileIn);
  matrixMul(s, setupResults);
  matMul_finish(s, fileIn, setupResults);

  //sync stream
  cudaStreamSynchronize(s);

  cudaStreamDestroy(s);
  
  return 0;
}
