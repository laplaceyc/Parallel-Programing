NVCC     := nvcc
NVFLAGS  := -arch=sm_20 -O3

all: sobel

clean:
	rm -f sobel
sobel: sobel.cu
	$(NVCC) $(NVFLAGS) -o $@ $?
