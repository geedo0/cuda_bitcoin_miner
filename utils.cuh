#ifndef UTILS_H
#define UTILS_H
#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

typedef struct {
	bool nonce_found;
	uint32_t nonce;
} Nonce_result;

void initialize_nonce_result(Nonce_result *nr);

#endif
