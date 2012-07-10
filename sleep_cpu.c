#include <stdio.h>
#include <time.h>
#include <signal.h>

int __nsleep(const struct timespec *req, struct timespec *rem)
{
  struct timespec temp_rem;
  if(nanosleep(req,rem)==-1)
    __nsleep(rem,&temp_rem);
  else
    return 1;
}

int msleep(unsigned long milisec)
{
  struct timespec req={0},rem={0};
  time_t sec=(int)(milisec/1000);
  milisec=milisec-(sec*1000);
  req.tv_sec=sec;
  req.tv_nsec=milisec*1000000L;
  __nsleep(&req,&rem);
  return 1;
}

void *sleep_cpu(char *filename)
{
    //open file
    FILE * ftp;
    ftp = fopen(filename,"r");

    // read the kernel_time from the file
    unsigned long *kernel_time = (unsigned long *) malloc(sizeof(int));
    fscanf(ftp, "%lu", kernel_time);

    fclose(ftp);

    msleep(*kernel_time);

    return (void *) kernel_time;
}


