#include <pthread.h>

//This method will let the kernel deallocate all the memory that it acquired in
//  kernel_setup and also lets the kernel write to its output file.
void kernel_finish(int kernel, cudaStream_t stream, char * filename, void *setupResult )
{
    if(kernel==1) sleep_finish(stream, filename, setupResult);

    if(kernel==2) matMul_finish(stream, filename, setupResult);
}


struct CleanerRecord
{
    int BatchSize;
    cudaStream_t *Streams;
    char **InputFiles;
    char **OutputFiles;
    int *Kernels;
    void **SetupResults;
    pthread_mutex_t Lock;
};

CleanerRecord *makeCleanerRecord(int batchSize, cudaStream_t *streams, char **inputFiles, 
                                    char **outputFiles, int *kernels, void **setupResults, pthread_mutex_t lock)
{
    CleanerRecord *r = (CleanerRecord *) malloc (sizeof(struct CleanerRecord));
    r->BatchSize = batchSize;
    r->Streams = streams;
    r->InputFiles = inputFiles;
    r->OutputFiles = outputFiles;
    r->Kernels = kernels;
    r->SetupResults = setupResults;
    r->Lock = lock;
    return r;
}



///////////////////////////////////////////////////////////////////////
// Cleaners's main
///////////////////////////////////////////////////////////////////////

void *cleaner_Main(void *params)
{
//   FOR ALL MY COMMENTS q :: 1..batchSize
//open up params (its actually a CleanerRecord *)
    CleanerRecord *r = (CleanerRecord *) params;

    int batchSize = r->BatchSize;
    cudaStream_t *streams = r->Streams;
    char **inputFiles = r->InputFiles;
    char **outputFiles = r->OutputFiles;
    int *kernels = r->Kernels;
    void **setupResults = r->SetupResults;
    pthread_mutex_t lock = r->Lock;

    pthread_mutex_lock(&lock);

//call kernel_finish for each kernel
    for(int q=0; q<batchSize; q++){
        kernel_finish(kernels[q], streams[q], outputFiles[q], setupResults[q]);
    }

//Synchronize with each streams[q]
    for(int q=0; q<batchSize; q++) cudaStreamSynchronize(streams[q]);

//deallocate

    for(int q=0;q<batchSize;q++){
        free(inputFiles[q]);
        free(outputFiles[q]);
        cudaStreamDestroy(streams[q]);
    }
    pthread_mutex_unlock(&lock);
	//free the arrays that we used;
    free(streams);
    free(kernels);
    free(inputFiles);
    free(outputFiles);
    free(setupResults);

    printf("cleaner finished a batch of kernels\n");
    return 0;
}






