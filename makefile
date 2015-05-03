OBJECTS=gpu_miner cpu_miner

all: $(OBJECTS)

clean:
	rm $(OBJECTS) sha256.o utils.o

gpu_miner: main.cu utils.o sha256.o
	nvcc -O1 -v -lrt -lm -arch=sm_20 -o $@ $^

verify_gpu: main.cu utils.o sha256.o
	nvcc -O1 -v -lrt -lm -D VERIFY_HASH -arch=sm_20 -o $@ $^

cpu_miner: serial_baseline.c sha256.o utils.o
	gcc -O1 -v -o $@ $^ -lrt

sha256.o: sha256.c
	gcc -O1 -v -c -o $@ $^

utils.o: utils.c
	gcc -O1 -v -c -o $@ $^ -lrt