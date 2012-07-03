//============================================
#include<stdio.h>
#include<stdlib.h>

//#define ROW 5
//#define COLUMN 5

void writeMatrixtoFile(int row, int column);

//------------------------------------------------------------------------------
//Main function

int main(int argc, char **argv)
{
  int ROW=0;
  int COLUMN =0;
  if( argc>1 ){
    ROW = atoi(argv[1]);       //could be used to pass in parameters
    COLUMN = atoi(argv[1]);       //could be used to pass in parameters       
  }
  printf("side length is %d\n", ROW);
  writeMatrixtoFile(ROW, COLUMN);

  return 0;
}

//------------------------------------------------------------------------------
//This function is for writing a matrix into a file
void writeMatrixtoFile(int row, int column)
{
  FILE *matrix=fopen("matrixIn.txt", "w");
  int a=0, b=0;
  for(a=0;a<row;a++)     
    {
      for(b=0;b<column;b++)  
        {              
	  fprintf(matrix, "%f\t", ((float)rand())/((float) RAND_MAX) );
        }
      fprintf(matrix, "\n");
    } 
      
  fclose(matrix);
}
