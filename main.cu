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

/*
	Threads = BDIMX*GDIMX*GDIMY
	Thread Max = 2^32
	512, 2^15, 2^8 fits everything
*/
#define BDIMX		512			//MAX = 512
#define GDIMX		65535		//MAX = 65535 = 2^16-1
#define GDIMY		1

__global__ void kernel_sha256d(SHA256_CTX *ctx, Nonce_result *nr, void *debug);

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

void compute_and_print_hash(unsigned char *data, unsigned int nonce) {
	unsigned char hash[32];
	SHA256_CTX ctx;
	int i;
	
	sha256_init(&ctx);
	sha256_update(&ctx, data, 80);
	sha256_final(&ctx, hash);
	sha256_init(&ctx);
	sha256_update(&ctx, hash, 80);
	sha256_final(&ctx, hash);

	printf("Hash is:\n");
	for(i=0; i<8; i++) {
		printf("%.8x ", ENDIAN_SWAP_32(*(((unsigned int *) hash) + i)));
	}
	printf("\n");
}

int main(int argc, char **argv) {
	int i, j;
	unsigned char *data = test_block;


	/*
		Host Side Preprocessing
		The goal here is to prepare and compute everything that will be shared by all threads.
	*/
	
	cudaPrintfInit();

	//Used to store a nonce if a block is mined
	Nonce_result h_nr;
	initialize_nonce_result(&h_nr);

	//Compute the shared portion of the SHA-256d calculation
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, (unsigned char *) data, 80);	//ctx.state contains a-h
	sha256_pad(&ctx);	

	//Decodes and stores the difficulty in a 32-byte array for convenience
	unsigned int nBits = ENDIAN_SWAP_32(*((unsigned int *) (data + 72)));
	set_difficulty(ctx.difficulty, nBits);								//ctx.data contains padded data

	//Data buffer for sending debug information to/from the GPU
	unsigned char debug[32];
	unsigned char *d_debug;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_debug, 32*sizeof(unsigned char)));
	CUDA_SAFE_CALL(cudaMemcpy(d_debug, (void *) &debug, 32*sizeof(unsigned char), cudaMemcpyHostToDevice));
	
	//Allocate space on Global Memory
	SHA256_CTX *d_ctx;
	Nonce_result *d_nr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_ctx, sizeof(SHA256_CTX)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_nr, sizeof(Nonce_result)));

	/*
		Kernel Execution
		Measure and launch the kernel and start mining
	*/
	//Copy data to device
	CUDA_SAFE_CALL(cudaMemcpy(d_ctx, (void *) &ctx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_nr, (void *) &h_nr, sizeof(Nonce_result), cudaMemcpyHostToDevice));

	//Launch kernel
	dim3 DimBlock(BDIMX,1);
	dim3 DimGrid(GDIMX,GDIMY);
	kernel_sha256d<<<DimGrid, DimBlock>>>(d_ctx, d_nr, (void *) d_debug);

	//Copy nonce result back to host
	CUDA_SAFE_CALL(cudaMemcpy((void *) &h_nr, d_nr, sizeof(Nonce_result), cudaMemcpyDeviceToHost));

	/*
		Post Processing
		Check the results of mining and print out debug information
	*/

	//Cuda Printf output
	cudaDeviceSynchronize();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();

	//Free memory on device
	CUDA_SAFE_CALL(cudaFree(d_ctx));
	CUDA_SAFE_CALL(cudaFree(d_nr));
	CUDA_SAFE_CALL(cudaFree(d_debug));

	if(h_nr.nonce_found) {
		printf("Nonce found! %.8x\n", h_nr.nonce);
		compute_and_print_hash(data, h_nr.nonce);
	}
	else {
		printf("Nonce not found :(\n");
	}

	return 0;
}

//Declare SHA-256 constants
__constant__ uint32_t k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__constant__ uint32_t h[8] = {
	0x6a09e667,	0xbb67ae85,	0x3c6ef372,	0xa54ff53a,
	0x510e527f,	0x9b05688c,	0x1f83d9ab,	0x5be0cd19
};

//Threads are organized mostly linearly but maximum GridDim forces some rows to exist
#define NONCE_VAL (gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x)

__global__ void kernel_sha256d(SHA256_CTX *ctx, Nonce_result *nr, void *debug) {
	unsigned int *mm = (unsigned int *) debug;
	unsigned int m[64];
	unsigned int hh[8];
	unsigned int a,b,c,d,e,f,g,h,t1,t2;
	int i, j;
	unsigned int nonce = NONCE_VAL;

	//Compute SHA-256 Message Schedule
	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (ctx->data[j] << 24) | (ctx->data[j + 1] << 16) | (ctx->data[j + 2] << 8) | (ctx->data[j + 3]);
	//Replace the nonce
	m[3] = nonce;
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	//Copy Initial Values into registers
	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	//Prepare input for next SHA-256
	m[0] = a + ctx->state[0];
	m[1] = b + ctx->state[1];
	m[2] = c + ctx->state[2];
	m[3] = d + ctx->state[3];
	m[4] = e + ctx->state[4];
	m[5] = f + ctx->state[5];
	m[6] = g + ctx->state[6];
	m[7] = h + ctx->state[7];
	//Pad the input
	m[8] = 0x80000000;	
	for(i=9; i<15; i++)
		m[i] = 0x00;
	m[15] = 0x00000100;	//Write out l=256
	for (i=16 ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	//Initialize the SHA-256 registers
	a = 0x6a09e667;
	b = 0xbb67ae85;
	c = 0x3c6ef372;
	d = 0xa54ff53a;
	e = 0x510e527f;
	f = 0x9b05688c;
	g = 0x1f83d9ab;
	h = 0x5be0cd19;
	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	hh[0] = ENDIAN_SWAP_32(a + 0x6a09e667);
	hh[1] = ENDIAN_SWAP_32(b + 0xbb67ae85);
	hh[2] = ENDIAN_SWAP_32(c + 0x3c6ef372);
	hh[3] = ENDIAN_SWAP_32(d + 0xa54ff53a);
	hh[4] = ENDIAN_SWAP_32(e + 0x510e527f);
	hh[5] = ENDIAN_SWAP_32(f + 0x9b05688c);
	hh[6] = ENDIAN_SWAP_32(g + 0x1f83d9ab);
	hh[7] = ENDIAN_SWAP_32(h + 0x5be0cd19);

	//cuPrintf("%.8x\n",hh[0]);
	unsigned char *hhh = (unsigned char *) hh;
	i=0;
	while(hhh[i] == ctx->difficulty[i])
		i++;

	if(hhh[i] < ctx->difficulty[i]) {
		//Synchronization Issue
		//Kind of a hack but it really doesn't matter which nonce
		//is written to the output, they're all winners :)
		//Further it's unlikely to even find a nonce let alone 2
		nr->nonce_found = true;
		nr->nonce = nonce;
	}
}