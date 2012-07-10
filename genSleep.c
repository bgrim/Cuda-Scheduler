//============================================
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
//------------------------------------------------------------------------------
//Main function

int main(int argc, char **argv)
{
  int sleepTime=32;
  char * filename = "sleepIn.txt";
  if( argc>1 ){
    sleepTime = atoi(argv[1]);  
  }
  if( argc>2) filename = argv[2];

  FILE *f=fopen(filename, "w");
  fprintf(f, "%d",sleepTime);
  fclose(f);

  printf("finished making:  %s\n", filename);
  return 0;
}
