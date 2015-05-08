#ifndef UTILS_H
#define UTILS_H

#include <time.h>

#define GIG 1000000000

extern struct timespec time1, time2;

#define tick()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1)
#define tock()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2)
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void print_execution_time();

bool reference_sha256(void *digest, const void *buffer, size_t length);
bool my_sha256(void *digest, const void *buffer, size_t length);

void verify_hash(bool my_sha(void*, const void*, size_t), unsigned char *data);
void output_block_header(char *filename, unsigned char *data, int length);

#endif
