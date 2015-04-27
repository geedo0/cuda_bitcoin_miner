#include "utils.h"

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