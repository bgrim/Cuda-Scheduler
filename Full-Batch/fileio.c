#include <stdio.h>   /* required for file operations */
FILE *fr;            /* declare the file pointer */

main()

{
  int n;
  long elapsed_seconds;
  char line[80];

  fr = fopen ("elapsed.dta", "rt");
  while(fgets(line, 80, fr) != NULL)
    {
      /* get a line, up to 80 chars from fr.  done if NULL */
      sscanf (line, "%ld", &elapsed_seconds);
      /* convert the string to a long int */
      printf ("%ld\n", elapsed_seconds);
    }
  fclose(fr);  /* close the file prior to exiting the routine */
} /*of main*/
