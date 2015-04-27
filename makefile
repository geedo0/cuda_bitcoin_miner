OBJECTS= gpu_miner
CC=nvcc

all: $(OBJECTS)

gpu_miner: main.cu utils.c sha256.c
	$(CC) -O1 -v -lrt -lm -arch=sm_20 -o $@ $^