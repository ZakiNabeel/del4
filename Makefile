######################################################################
# Choose your favorite C compiler
CC = gcc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2)


######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o

all:  lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

example1: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt

######################################################################
# CUDA (optional) — appended section; original targets unchanged

# Toolchains
NVCC       ?= nvcc
CXX        ?= g++
# Set your architecture if you know it (e.g., sm_86 for RTX 30xx, sm_80 for A100)
# Build for a set of common architectures (P100, V100, T4, A100, RTX30, L4)
CUDAARCHES ?= 60 70 75 80 86 89
GENCODES   := $(foreach a,$(CUDAARCHES),-gencode arch=compute_$(a),code=sm_$(a))

CUDAFLAGS  ?= -O3 -Xcompiler -fPIC -I. -I./cuda $(GENCODES)


# Flags
CUDAFLAGS  ?= -O3 -arch=$(CUDAARCH) -Xcompiler -fPIC -I. -I./cuda
CUDALIBS   ?= -lcudart
# You can add -DUSE_CUDA here or in your compile line below
CXXFLAGS   ?=

# CUDA sources (create these files as you implement the GPU path)
# CUDA sources
CUDA_SRCS  := cuda/pyramid_cuda.cu \
              cuda/gradients_cuda.cu \
              cuda/track_cuda.cu \
              cuda/example_cuda.cu

CUDA_OBJS  := $(CUDA_SRCS:.cu=.o)

cuda/%.o: cuda/%.cu
	$(NVCC) $(CUDAFLAGS) -I. -I./cuda -c $< -o $@

$(CUDA_LIB): $(CUDA_OBJS)
	@rm -f $@
	ar rcs $@ $(CUDA_OBJS)


CUDA_LIB   := libklt_cuda.a

# Build CUDA objects
cuda/%.o: cuda/%.cu cuda/klt_cuda.h
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Pack CUDA objects into a static lib (like your CPU libklt.a)
$(CUDA_LIB): $(CUDA_OBJS)
	@rm -f $@
	ar rcs $@ $(CUDA_OBJS)

# Unified runner that can call CPU examples or the CUDA path (uses main.cpp)
# Build with USE_CUDA so the GPU code paths are compiled in.
# GPU-only runner that does NOT link example1..5 (avoids multiple mains)
klt_runner: libklt.a $(CUDA_LIB) main_gpu.cpp
	$(NVCC) $(CUDAFLAGS) -DUSE_CUDA -o $@ main_gpu.cpp \
		-L. -lklt -lklt_cuda $(LIB) -lcudart


# Convenience alias to build the GPU runner
gpu: klt_runner

# Extend clean to remove CUDA artifacts (keeps existing 'clean' behavior intact)
.PHONY: clean-cuda
clean-cuda:
	rm -f $(CUDA_OBJS) $(CUDA_LIB) klt_runner

########################################################################
# OpenACC Section (CPU vs GPU builds)
########################################################################

# Path to OpenACC compiler (NVHPC) – use from PATH by default
NVC ?= nvc

# Baseline CPU executable (OpenACC pragmas ignored)
cpu: libklt.a example3_cpu

example3_cpu: example3.c libklt.a
	$(CC) -O3 $(CFLAGS) -o example3_cpu example3.c -L. -lklt $(LIB) -lm

# OpenACC-accelerated executable
acc: example3_acc

example3_acc: example3.c $(ARCH)
	$(NVC) -O3 -acc -Minfo=accel $(CFLAGS) \
		-o example3_acc example3.c $(ARCH) $(LIB) -lm

########################################################################
# Timing helpers (simple, readable ms output)
########################################################################

time_cpu: cpu
	@echo "Running CPU version..."
	@start=$$(date +%s%N); \
	./example3_cpu; \
	end=$$(date +%s%N); \
	delta_ns=$$((end-start)); \
	ms=$$((delta_ns/1000000)); \
	echo "CPU time: $$ms ms"

time_acc: acc
	@echo "Running OpenACC version..."
	@start=$$(date +%s%N); \
	./example3_acc; \
	end=$$(date +%s%N); \
	delta_ns=$$((end-start)); \
	ms=$$((delta_ns/1000000)); \
	echo "ACC time: $$ms ms"
