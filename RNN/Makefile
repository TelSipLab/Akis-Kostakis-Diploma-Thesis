# Simple Makefile for RNN/LSTM examples

# Libtorch paths
LIBTORCH_PATH = /home/kostakis/dev/libtorch
LIBTORCH_INCLUDE = -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
LIBTORCH_LIBS = -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10
LIBTORCH_RPATH = -Wl,-rpath,$(LIBTORCH_PATH)/lib

CSVLIBRARY_INCLUDE = -I../include

# Compiler flags
CXX = g++
CXXFLAGS = -std=c++17 -g -O0 -D_GLIBCXX_USE_CXX11_ABI=1

all: example lstmExample lstm

example: mainExample.cpp
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) \
		mainExample.cpp -o example.out \
		$(LIBTORCH_LIBS) $(LIBTORCH_RPATH)

lstmExample: lstmExample.cpp
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) \
		lstmExample.cpp -o example.out \
		$(LIBTORCH_LIBS) $(LIBTORCH_RPATH)

lstm: lstmMain.cpp
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) $(CSVLIBRARY_INCLUDE) \
		lstmMain.cpp -o lstm.out \
		$(LIBTORCH_LIBS) $(LIBTORCH_RPATH)

clean:
	rm -f example.out lstmExample.out

.PHONY: all clean
