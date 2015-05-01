#include <stdio.h>
#include <stdlib.h>

#include "sha256.h"
#include "utils.h"
#include "test.h"

#define NUM_EXPERIMENTS		7

//Define when you want to break out of the loop when the block is solved.
//#define MINING_MODE

int main(int argc, char *argv) {
	int i, j, k;
	unsigned char *data = test_block;
	unsigned char hash[32], difficulty[32];
	SHA256_CTX ctx;
	Nonce_result nr;

	initialize_nonce_result(&nr);

	unsigned int nBits = ENDIAN_SWAP_32(*((unsigned int *) (data + 72)));
	set_difficulty(difficulty, nBits);

	int hashes = 1;
	for(i=0; i<32; i++) {
		tick();
		for(j=0; j<hashes; j++) {
			//Hash the block header
			sha256_init(&ctx);
			sha256_update(&ctx, data, 80);
			sha256_final(&ctx, hash);
			//Hash
			sha256_init(&ctx);
			sha256_update(&ctx, hash, 32);
			sha256_final(&ctx, hash);

			//Check the difficulty
			k=0;
			while(hash[k] == difficulty[k]) k++;
			if(hash[k] < difficulty[k]) {
				nr.nonce_found = true;
				nr.nonce = j;
				#ifdef MINING_MODE
				break;
				#endif
			}
		}
		tock();
		//Print hashes, execution time
		printf("%d,%ld\n",hashes,get_execution_time());
		hashes <<= 1;
	}
	if(nr.nonce_found) {
		printf("Nonce found! %.8x\n", nr.nonce);
	}
	else {
		printf("Nonce not found :(\n");
	}

}