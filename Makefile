CXX_STANDARD = c++14

all: precompile main

main: precompile
	g++ *.cpp -std=$(CXX_STANDARD) -o main.out -g

run: main
	./main.out

precompile: pch.h.gch

pch.h.gch: pch.h
	g++ -x c++-header pch.h -o pch.h.gch

clean:
	rm -f main.out pch.h.gch
