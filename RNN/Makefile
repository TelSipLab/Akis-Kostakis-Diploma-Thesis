# Simple Makefile for RNN example

all: example

example: mainExample.cpp
	g++ -std=c++17 -g -O0 -D_GLIBCXX_USE_CXX11_ABI=1 \
		-I/home/kostakis/dev/libtorch/include \
		-I/home/kostakis/dev/libtorch/include/torch/csrc/api/include \
		mainExample.cpp -o example.out \
		-L/home/kostakis/dev/libtorch/lib -ltorch -ltorch_cpu -lc10 \
		-Wl,-rpath,/home/kostakis/dev/libtorch/lib
clean:
	rm -f example.out

.PHONY: clean
