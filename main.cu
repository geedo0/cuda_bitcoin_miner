#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

#include "utils.cuh"

#include "test.h"

int main(int argc, char **argv) {
	int i;
	unsigned char *data = test_block;
	size_t datasz = 76;

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
