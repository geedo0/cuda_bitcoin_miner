#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

#include "test.h"

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true);
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef struct {
	bool nonce_found;
	uint32_t nonce;
} Nonce_result;

void initialize_nonce_result(Nonce_result *nr) {
	nr->nonce_found = true;
	nr->nonce = 0xdeadbeef;
}

int main(int argc, char **argv) {
	int i;
	unsigned char *data = test_block;
	size_t datasz = 76;

	for(i=0; i<test_block_length; i++) {
		if(!(i%10))
			printf("\n");
		printf("0x%.2hhx, ", data[i]);
	}

	//Initialize Nonce Result to see if we solved the block
	Nonce_result nr;
	initialize_nonce_result(&nr);
	
	//Allocate space on Global Memory
	unsigned char *d_block_header;
	Nonce_result *d_nr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_block_header, datasz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));

	//Copy data structures to device
	CUDA_SAFE_CALL(cudaMemcpy(d_block_header, data, datasz, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

	//Copy data back to host
	Nonce_result nr_from_device;
	CUDA_SAFE_CALL(cudaMemcpy((void *) &nr_from_device, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

	if(nr_from_device.nonce_found) {
		printf("Nonce found! %.8x\n", nr_from_device.nonce);
	}
	else {
		printf("Nonce not found :(\n");
	}
	

	if(nr_from_device.nonce == nr.nonce)
		printf("Copy passed\n");
	else
		printf("Copy failed %d %d", nr_from_device.nonce, nr.nonce);

	return 0;
}

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}