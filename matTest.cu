#include "matrixMul_kernel.cu"
#include <stdio.h>

//headers

//main
int main(){
  
  int jobs = 25600;
  int throttle = 1;

  // create streams as many throttle
  cudaStream_t* c = (cudaStream_t* )malloc(throttle*sizeof(cudaStream_t));
  int i;
  for(i=0;i<throttle;i++){
    cudaStreamCreate(&c[i]);
  }

  // set input file
  char *fileIn = "Inputs/matrixIn0.txt";;

  // do work for each job
  int k;
  void* setupResults = matMul_setup(c[0], fileIn);
  cudaDeviceSynchronize();

  for(k=0;k<jobs;k++){
    matrixMul(c[k%throttle], setupResults);
  }
  matMul_finish(c[0], fileIn, setupResults);  

  // sync device
  cudaDeviceSynchronize();

  // destroy streams
  int j;
  for(j=0;j<throttle;j++){
    cudaStreamDestroy(c[j]);
  }
  
  // free array
  free(c);

  // return
  return 0;
}
