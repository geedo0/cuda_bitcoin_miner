#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#define GIG 1000000000

extern struct timespec time1, time2;

#define tick()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1)
#define tock()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2)

#define ENDIAN_SWAP_32(x) (\
	((x & 0xff000000) >> 24) | \
	((x & 0x00ff0000) >> 8 ) | \
	((x & 0x0000ff00) << 8 ) | \
	((x & 0x000000ff) << 24))

typedef struct {
	bool nonce_found;
	uint32_t nonce;
} Nonce_result;

long int get_execution_time();
void initialize_nonce_result(Nonce_result *nr);
void set_difficulty(unsigned char *difficulty, unsigned int nBits);

#endif
