#include<stdio.h>

int getMatrixSideLengthFromFile(char* filename){
  int c=1;
  char ch='\0';
  FILE* ftp;
  ftp=fopen(filename, "r");
  while(ch!='\n') {
    ch=fgetc(ftp);
    if(ch=='\t'){
      c++;
    }
  }
  return c;
}


main()
{
  int lines = getMatrixSideLengthFromFile("matrix.txt");
  printf("Lines are: %d\n", lines);
}
