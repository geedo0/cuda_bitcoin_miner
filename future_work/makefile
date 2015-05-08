OBJECTS=my_miner utils.o sha256.o
INCLUDES=-I/usr/local/include/libblkmaker-0.1
LIBRARY_PATHS=-L/usr/local/lib/
LIBRARIES=-lbase58 -lblkmaker-0.1 -ljansson -lgcrypt -lblkmaker_jansson-0.1 -lrt
FLAGS=-std=gnu99 -v -g

all: $(OBJECTS)

clean:
	rm $(OBJECTS)

my_miner: main.c utils.o sha256.o
	gcc $(FLAGS) $(INCLUDES) $(LIBRARY_PATHS) -o $@ $^ $(LIBRARIES)

utils.o: utils.c
	gcc -std=gnu99 -O1 -c -o $@ $^ -lgcrypt -lrt

sha256.o: sha256.c
	gcc -O1 -c -o $@ $^