# Simple Makefile for RNN/LSTM examples

# UPDATE with the directory you unzipped libtorch
LIBTORCH_PATH = /home/kostakis/dev/libtorch

LIBTORCH_INCLUDE = -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
UTILS_LIBRARY_INCLUDE = -Iinclude

LIBTORCH_LIBS = -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10
LIBTORCH_RPATH = -Wl,-rpath,$(LIBTORCH_PATH)/lib

# Compiler flags
CXX = g++
CXXFLAGS = -std=c++17 -g -Wall -O0 -D_GLIBCXX_USE_CXX11_ABI=1

all: example lstmExample lstm lstmEval

example: mainExample.cpp
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) \
		mainExample.cpp -o example.out \
		$(LIBTORCH_LIBS) $(LIBTORCH_RPATH)

lstmExample: lstmExample.cpp
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) \
		lstmExample.cpp -o example.out \
		$(LIBTORCH_LIBS) $(LIBTORCH_RPATH)

lstm: lstmMain.cpp
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) $(UTILS_LIBRARY_INCLUDE) \
		lstmMain.cpp -o lstm.out \
		$(LIBTORCH_LIBS) $(LIBTORCH_RPATH)

lstmEval: lstmEval.cpp
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) $(UTILS_LIBRARY_INCLUDE) \
		lstmEval.cpp -o lstmEval.out \
		$(LIBTORCH_LIBS) $(LIBTORCH_RPATH)

clean:
	rm -f *.out *.o

.PHONY: all clean
