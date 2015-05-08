#include <stdio.h>
#include <stdbool.h>
#include <gcrypt.h>

#include "utils.h"
#include "sha256.h"

//Forward declarations for "private" functions
struct timespec diff(struct timespec start, struct timespec end);
bool reference_sha256(void *digest, const void *buffer, size_t length);

struct timespec time1, time2;

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

void print_execution_time() {
	struct timespec delta = diff(time1,time2);
	printf("\n%ld ns elapsed.\n", (long int) (GIG * delta.tv_sec + delta.tv_nsec));
}

bool reference_sha256(void *digest, const void *buffer, size_t length)
{
  gcry_md_hash_buffer(GCRY_MD_SHA256, digest, buffer, length);
  return true;
}

bool my_sha256(void *digest, const void *buffer, size_t length) {
  SHA256_CTX ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, (unsigned char *) buffer, length);
  sha256_final(&ctx, (unsigned char *) digest);
  return true;
}

void verify_hash(bool my_sha(void*, const void*, size_t), unsigned char *data)
{
  unsigned char reference_hash[32];
  unsigned char my_hash[32];

  //Compute reference hash
  tick();
  reference_sha256(reference_hash, data, (size_t) 80);
  reference_sha256(reference_hash, reference_hash, (size_t) 32);
  tock();
  print_execution_time();

  //Compute my hash
  tick();
  my_sha(my_hash, data, (size_t) 80);
  my_sha(my_hash, my_hash, (size_t) 32);
  tock();
  print_execution_time();

  //Compare
  for(int i=0; i<32; i++) {
    if(reference_hash[i] != my_hash[i]) {
      printf("\nHash verification failed\n");
      return;
    }
  }
  printf("\nHash verification passed\n");
}

void output_block_header(char *filename, unsigned char *data, int length) {
  int i;
  FILE *fd = fopen(filename, "w+");
  fprintf(fd, "const unsigned char test_block[] = {");
  for(i=0; i<length-1; i++) {
    fprintf(fd,"0x%.2hhx, ",data[i]);
  }
  fprintf(fd, "0x%.2hhx};\nint test_block_length = %d;\n", data[length-1], length);
  fclose(fd);
}