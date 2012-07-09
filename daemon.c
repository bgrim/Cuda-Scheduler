#include "queue.c"
#include <pthread.h>

//This file contains functions used to run and interact with the daemon
//currently it doesn't interact with the outside world to determine the
//jobs and input files it needs to use

///////////////////////////////////////////////////////////////////////
//  Global variables
///////////////////////////////////////////////////////////////////////

//WARNING: The following three queues all contain (void *) and return (void *)
//         This must be done to allow them to have different types:
//           kernelTypes will return (void *) that are actually (int *)
//           inputFiles will return (void *) that are actually (char **)
//           outputFiles will return (void *) that are actually (char **)
Queue kernelTypes;
Queue inputFiles;
Queue outputFiles;


pthread_mutex_t queueLock;

///////////////////////////////////////////////////////////////////////
//  Functions to interact with the Daemon
///////////////////////////////////////////////////////////////////////

void daemon_init(int maxJobs)
{
  if(maxJobs<5) maxJobs=5;
  kernelTypes = CreateQueue(maxJobs);
  inputFiles = CreateQueue(maxJobs);
  outputFiles = CreateQueue(maxJobs);
  pthread_mutex_init(&queueLock, NULL);
}

void daemon_free()
{
  DisposeQueue(kernelTypes);
  DisposeQueue(inputFiles);
  DisposeQueue(outputFiles);
  pthread_mutex_destroy(&queueLock);
}

int daemon_GetNextKernel()
{
  bool noResult = true;
  void *result;
  while(noResult){
    pthread_mutex_lock(&queueLock);

    noResult = IsEmpty(kernelTypes);
    if( !noResult ) result = FrontAndDequeue(kernelTypes);

    pthread_mutex_unlock(&queueLock);
    if(noResult) pthread_yield();
  }
  return *((int *) result);
}

char *daemon_GetInputFile()
{
  bool noResult = true;
  void *result;
  while(noResult){
    pthread_mutex_lock(&queueLock);

    noResult = IsEmpty(inputFiles);
    if( !noResult ) result = FrontAndDequeue(inputFiles);

    pthread_mutex_unlock(&queueLock);
    if(noResult) pthread_yield();
  }
  return (char *) result;
}

char *daemon_GetOutputFile()
{
  bool noResult = true;
  void *result;
  while(noResult){
    pthread_mutex_lock(&queueLock);

    noResult = IsEmpty(outputFiles);
    if( !noResult ) result = FrontAndDequeue(outputFiles);

    pthread_mutex_unlock(&queueLock);
    if(noResult) pthread_yield();
  }
  return (char *) result;
}



///////////////////////////////////////////////////////////////////////
// Daemon's main and any helper functions for it
///////////////////////////////////////////////////////////////////////

void *daemon_Main(void *numOfJobs)
{
  int max=64;  //the lasrgest number of chars in a filename

  int *jobs = (int *) numOfJobs;
  for(int i=0; i<(*jobs); i++){
    pthread_mutex_lock(&queueLock);

    //Currently these three values are hardcoded. Evntually the daemon will
    //  interact with swift or a similar program to determine what to run.
    //These will be deallocated in Scheduler.cu.
    int *kernelType = (int *) malloc(sizeof(int));
    *kernelType = 2;

    char *fileIn = (char *) malloc(sizeof(char)*(max+1));
    // sprintf(fileIn, "Inputs/matrixIn%d.txt", i);
    // sprintf(fileIn, "/dev/shm/Inputs/matrixIn%d.txt", i);
    sprintf(fileIn, "Inputs/matrixIn%d.txt", i);

    char *fileOut = (char *) malloc(sizeof(char)*(max+1));
    // sprintf(fileOut, "Outputs/matrixOut%d.txt", i);
    // sprintf(fileOut, "/dev/shm/Outputs/matrixOut%d.txt", i);
    sprintf(fileOut, "Outputs/matrixOut%d.txt", i);

    Enqueue((void *) kernelType, kernelTypes);
    Enqueue((void *) fileIn, inputFiles); //could make the file name depend on i
    Enqueue((void *) fileOut, outputFiles); //also could depend on i
    
    pthread_mutex_unlock(&queueLock);
  }
  printf("daemon finished loading jobs\n\n");
  return 0;
}
