#include "utils.h"

struct timespec diff(struct timespec start, struct timespec end);

struct timespec time1, time2;

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

long int get_execution_time() {
	struct timespec delta = diff(time1,time2);
	return (long int) (GIG * delta.tv_sec + delta.tv_nsec);
}

void initialize_nonce_result(Nonce_result *nr) {
	nr->nonce_found = false;
	nr->nonce = 0;
}

//difficulty MUST be 32 bytes
void set_difficulty(unsigned char *difficulty, unsigned int nBits) {
	int i;
	for(i=0; i<32; i++) {
		difficulty[i] = 0;
	}
	int msb = 32 - ((nBits & 0xff000000) >> 24);
	difficulty[msb++] = (nBits & 0xff0000) >> 16;
	difficulty[msb++] = (nBits & 0xff00) >> 8;
	difficulty[msb] = nBits & 0xff;
}