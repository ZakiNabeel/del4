######################################################################
# Makefile for Del4 - OpenACC Optimization Project
######################################################################

# Compilers
CC = gcc
NVC ?= nvc

# Directories
CPU_SRC_CORE = cpu/src/core
CPU_SRC_FEATURES = cpu/src/features
CPU_SRC_IO = cpu/src/io
CPU_INCLUDE = cpu/include

GPU_SRC_CORE = gpu/src/core
GPU_SRC_FEATURES = gpu/src/features
GPU_SRC_IO = gpu/src/io
GPU_INCLUDE = gpu/include

EXAMPLES = examples
BUILD = build
OUTPUT = output
DATA_DIR ?= data/pitch/frames

# Create output directories
$(shell mkdir -p $(BUILD) $(OUTPUT)/cpu/frames $(OUTPUT)/gpu/frames)

######################################################################
# Flags
######################################################################
FLAG1 = -DNDEBUG
CPU_CFLAGS = $(FLAG1) -I$(CPU_INCLUDE) -O3
GPU_CFLAGS = $(FLAG1) -I$(GPU_INCLUDE) -O3
ACC_FLAGS = -acc -gpu=managed -Minfo=accel -O3 -I$(GPU_INCLUDE) $(FLAG1)

LIB = -L/usr/local/lib -L/usr/lib

######################################################################
# Source Files
######################################################################
CPU_CORE_SRCS = $(CPU_SRC_CORE)/convolve.c \
                $(CPU_SRC_CORE)/pyramid.c \
                $(CPU_SRC_CORE)/klt.c \
                $(CPU_SRC_CORE)/klt_util.c

CPU_FEAT_SRCS = $(CPU_SRC_FEATURES)/selectGoodFeatures.c \
                $(CPU_SRC_FEATURES)/storeFeatures.c \
                $(CPU_SRC_FEATURES)/trackFeatures.c \
                $(CPU_SRC_FEATURES)/writeFeatures.c

CPU_IO_SRCS = $(CPU_SRC_IO)/error.c \
              $(CPU_SRC_IO)/pnmio.c

CPU_SRCS = $(CPU_CORE_SRCS) $(CPU_FEAT_SRCS) $(CPU_IO_SRCS)
CPU_OBJS = $(patsubst %.c,$(BUILD)/cpu_%.o,$(notdir $(CPU_SRCS)))

# GPU/OpenACC sources (same structure)
GPU_CORE_SRCS = $(GPU_SRC_CORE)/convolve.c \
                $(GPU_SRC_CORE)/pyramid.c \
                $(GPU_SRC_CORE)/klt.c \
                $(GPU_SRC_CORE)/klt_util.c

GPU_FEAT_SRCS = $(GPU_SRC_FEATURES)/selectGoodFeatures.c \
                $(GPU_SRC_FEATURES)/storeFeatures.c \
                $(GPU_SRC_FEATURES)/trackFeatures.c \
                $(GPU_SRC_FEATURES)/writeFeatures.c

GPU_IO_SRCS = $(GPU_SRC_IO)/error.c \
              $(GPU_SRC_IO)/pnmio.c

GPU_SRCS = $(GPU_CORE_SRCS) $(GPU_FEAT_SRCS) $(GPU_IO_SRCS)
GPU_OBJS = $(patsubst %.c,$(BUILD)/gpu_%.o,$(notdir $(GPU_SRCS)))

######################################################################
# Default Target
######################################################################
all: cpu acc

######################################################################
# CPU Build (Baseline - No OpenACC)
######################################################################

# Compile CPU object files
$(BUILD)/cpu_convolve.o: $(CPU_SRC_CORE)/convolve.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_pyramid.o: $(CPU_SRC_CORE)/pyramid.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_klt.o: $(CPU_SRC_CORE)/klt.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_klt_util.o: $(CPU_SRC_CORE)/klt_util.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_selectGoodFeatures.o: $(CPU_SRC_FEATURES)/selectGoodFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_storeFeatures.o: $(CPU_SRC_FEATURES)/storeFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_trackFeatures.o: $(CPU_SRC_FEATURES)/trackFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_writeFeatures.o: $(CPU_SRC_FEATURES)/writeFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_error.o: $(CPU_SRC_IO)/error.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

$(BUILD)/cpu_pnmio.o: $(CPU_SRC_IO)/pnmio.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

# CPU library
libklt_cpu.a: $(CPU_OBJS)
	@rm -f $@
	ar ruv $@ $(CPU_OBJS)
	@echo "✅ CPU Library built: libklt_cpu.a"

# CPU executable
main_cpu: libklt_cpu.a $(EXAMPLES)/main_cpu.c
	$(CC) $(CPU_CFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(OUTPUT)/cpu/frames/"' \
		-o $@ $(EXAMPLES)/main_cpu.c -L. -lklt_cpu $(LIB) -lm
	@echo "✅ CPU executable built: main_cpu"

cpu: main_cpu

######################################################################
# GPU Build (OpenACC with NVC)
######################################################################

# Compile GPU object files with OpenACC
$(BUILD)/gpu_convolve.o: $(GPU_SRC_CORE)/convolve.c
	$(NVC) -c $(ACC_FLAGS) $< -o $@

$(BUILD)/gpu_pyramid.o: $(GPU_SRC_CORE)/pyramid.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_klt.o: $(GPU_SRC_CORE)/klt.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_klt_util.o: $(GPU_SRC_CORE)/klt_util.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_selectGoodFeatures.o: $(GPU_SRC_FEATURES)/selectGoodFeatures.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_storeFeatures.o: $(GPU_SRC_FEATURES)/storeFeatures.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_trackFeatures.o: $(GPU_SRC_FEATURES)/trackFeatures.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_writeFeatures.o: $(GPU_SRC_FEATURES)/writeFeatures.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_error.o: $(GPU_SRC_IO)/error.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

$(BUILD)/gpu_pnmio.o: $(GPU_SRC_IO)/pnmio.c
	$(NVC) -c $(GPU_CFLAGS) $< -o $@

# GPU library
libklt_gpu.a: $(GPU_OBJS)
	@rm -f $@
	ar ruv $@ $(GPU_OBJS)
	@echo "✅ GPU Library built: libklt_gpu.a"

# GPU executable
main_gpu: libklt_gpu.a $(EXAMPLES)/main_gpu.c
	$(NVC) $(ACC_FLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(OUTPUT)/gpu/frames/"' \
		-o $@ $(EXAMPLES)/main_gpu.c -L. -lklt_gpu $(LIB) -lm
	@echo "✅ GPU executable built: main_gpu"

acc: main_gpu
gpu: main_gpu

######################################################################
# Run Targets with Timing
######################################################################
run_cpu: cpu
	@echo "🚀 Running CPU version..."
	@./main_cpu

run_gpu: acc
	@echo "🚀 Running GPU version..."
	@./main_gpu

run_acc: run_gpu

# Timing with comparison
time_cpu: cpu
	@echo "⏱️  Timing CPU version..."
	@/usr/bin/time -f "CPU Time: %E (elapsed) %U (user) %S (sys)" ./main_cpu

time_gpu: acc
	@echo "⏱️  Timing GPU version..."
	@/usr/bin/time -f "GPU Time: %E (elapsed) %U (user) %S (sys)" ./main_gpu

time_acc: time_gpu

# Run both and compare
compare: cpu acc
	@echo "========================================="
	@echo "🔥 Performance Comparison"
	@echo "========================================="
	@echo ""
	@echo "Running CPU version..."
	@/usr/bin/time -f "CPU Time: %E" ./main_cpu 2>&1 | grep "CPU Time:"
	@echo ""
	@echo "Running GPU version..."
	@/usr/bin/time -f "GPU Time: %E" ./main_gpu 2>&1 | grep "GPU Time:"
	@echo ""
	@echo "========================================="

######################################################################
# Clean
######################################################################
clean:
	rm -f $(BUILD)/*.o *.a main_cpu main_gpu
	rm -f $(OUTPUT)/cpu/frames/*.ppm $(OUTPUT)/gpu/frames/*.ppm
	rm -f $(OUTPUT)/cpu/frames/*.txt $(OUTPUT)/gpu/frames/*.txt
	@echo "✅ Cleaned build artifacts"

clean-all: clean
	rm -rf $(BUILD) $(OUTPUT)
	@echo "✅ Cleaned everything"

######################################################################
# Help
######################################################################
help:
	@echo "Available targets:"
	@echo "  make cpu        - Build CPU version (baseline)"
	@echo "  make acc/gpu    - Build GPU version with OpenACC"
	@echo "  make all        - Build both CPU and GPU versions"
	@echo ""
	@echo "  make run_cpu    - Run CPU version"
	@echo "  make run_gpu    - Run GPU version"
	@echo ""
	@echo "  make time_cpu   - Time CPU version"
	@echo "  make time_gpu   - Time GPU version"
	@echo "  make compare    - Run both and compare times"
	@echo ""
	@echo "  make clean      - Clean build artifacts"
	@echo "  make clean-all  - Clean everything including output"

.PHONY: all cpu acc gpu run_cpu run_gpu run_acc time_cpu time_gpu time_acc compare clean clean-all help
