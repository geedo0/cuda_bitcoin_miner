#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

#include "cuPrintf.cu"
#include "cuPrintf.cuh"
extern "C" {
	#include "sha256.h"
	#include "utils.h"
}

#include "test.h"

__global__ void kernel_sha256d(SHA256_CTX *ctx, Nonce_result *nr);

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

int main(int argc, char **argv) {
	int i;
	unsigned char *data = test_block;

	//Compute the first SHA-256 message block
	//Registers a-h stored in ctx.state
	//Remainder data is padded and stored in ctx.data, just change the nonce
	SHA256_CTX ctx;
	sha256_init(&ctx);
	//Update leaves the remainder data in ctx
	sha256_update(&ctx, (unsigned char *) data, 80);
	sha256_pad(&ctx);

	cudaPrintfInit();

	//Initialize Nonce Result to see if we solved the block
	Nonce_result h_nr;
	initialize_nonce_result(&h_nr);
	
	//Allocate space on Global Memory
	SHA256_CTX *d_ctx;
	Nonce_result *d_nr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_ctx, sizeof(SHA256_CTX)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));

	//Copy data structures to device
	CUDA_SAFE_CALL(cudaMemcpy(d_ctx, (void *) &ctx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &h_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

	//Launch kernel
	dim3 DimBlock(1,1);
	dim3 DimGrid(1,1);
	kernel_sha256d<<<DimGrid, DimBlock>>>(d_ctx, d_nr);

	//Copy data back to host
	CUDA_SAFE_CALL(cudaMemcpy((void *) &h_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

	//Cuda Printf output
	cudaDeviceSynchronize();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();

	//Free memory on device
	CUDA_SAFE_CALL(cudaFree(d_ctx));
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