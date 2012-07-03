#include<stdio.h>

int getMatrixSideLengthFromFile(char* filename){
  int c=0;
  char ch='\0';
  FILE* ftp;
  ftp=fopen(filename, "r");
  while(ch!=EOF) {
    ch=fgetc(ftp);
    if(ch=='\n'){
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
