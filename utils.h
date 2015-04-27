#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
	bool nonce_found;
	uint32_t nonce;
} Nonce_result;

void initialize_nonce_result(Nonce_result *nr);
void set_difficulty(unsigned char *difficulty, unsigned int nBits);

#endif
