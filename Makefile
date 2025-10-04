CXX_STANDARD = c++14
INCLUDES = -Iinclude
SRC_DIR = src
BUILD_DIR = build

SOURCES = $(wildcard $(SRC_DIR)/*.cpp) main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: main

main: $(SOURCES)
	g++ $(SOURCES) $(INCLUDES) -std=$(CXX_STANDARD) -o main.out -g

run: main
	./main.out

clean:
	rm -f main.out $(SRC_DIR)/*.o *.o
