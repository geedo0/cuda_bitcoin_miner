#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

#include "cuPrintf.cu"
#include "cuPrintf.cuh"
#include "utils.cuh"
extern "C" {
#include "sha256.h"
}

#include "test.h"

__global__ void kernel_sha256d(unsigned char *data, Nonce_result *nr);

int main(int argc, char **argv) {
	int i;
	unsigned char *data = test_block;
	size_t datasz = 76;

	//Compute the first SHA-256 message block
	//Registers a-h stored in ctx.state
	//Remainder data is padded and stored in ctx.data, just change the nonce
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char *) data, 80);
	sha256_pad(&ctx);

	cudaPrintfInit();

	//Initialize Nonce Result to see if we solved the block
	Nonce_result h_nr;
	initialize_nonce_result(&h_nr);
	
	//Allocate space on Global Memory
	unsigned char *d_block_header;
	Nonce_result *d_nr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_block_header, datasz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));

	//Copy data structures to device
	CUDA_SAFE_CALL(cudaMemcpy(d_block_header, data, datasz, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &h_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

	//Launch kernel
	dim3 DimBlock(1,1);
	dim3 DimGrid(1,1);
	kernel_sha256d<<<DimGrid, DimBlock>>>(d_block_header, d_nr);

	//Copy data back to host
	CUDA_SAFE_CALL(cudaMemcpy((void *) &h_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

	//Cuda Printf output
	cudaDeviceSynchronize();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();

	//Free memory on device
	CUDA_SAFE_CALL(cudaFree(d_block_header));
	CUDA_SAFE_CALL(cudaFree(d_nr));

	if(h_nr.nonce_found) {
		printf("Nonce found! %.8x\n", h_nr.nonce);
	}
	else {
		printf("Nonce not found :(\n");
	}

	return 0;
}

__global__ void kernel_sha256d(unsigned char *data, Nonce_result *nr) {
	cuPrintf("Hello nonce %x\n", nr->nonce);
	nr->nonce = 0xdeadbeef;
	nr->nonce_found = true;
}