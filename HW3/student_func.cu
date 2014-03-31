/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/



#include "reference_calc.cpp"
#include "utils.h"
#include <float.h>

__device__ static float atomicMaxf(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMinf(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_reduce(const float* const d_array, float* d_max, 
                                              const size_t elements)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = FLT_MIN;

    // load shared memory from global memory
    if (gid < elements)
        shared[tid] = d_array[gid];
    __syncthreads();

    // do max reduction in shared memory
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    
    // only thread 0 writes result for this block back to global memory
    if (tid == 0)
      atomicMaxf(d_max, shared[0]);
}

__global__ void min_reduce(const float* const d_array, float* d_min, 
                                              const size_t elements)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = FLT_MAX;

    // load shared memory from global memory
    if (gid < elements)
        shared[tid] = d_array[gid];
    __syncthreads();

    // do min reduction in shared memory
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
            shared[tid] = min(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    
    // only thread 0 writes result for this block back to global memory
    if (tid == 0)
      atomicMinf(d_min, shared[0]);
}

__global__
void histogram(const float* const d_logLuminance, 
							unsigned int* histo,
							float logLumMin,
							float logLumRange,
                            const size_t numBins)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	// formula: bin = (lum[i] - lumMin) / lumRange * numBins
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1), static_cast<unsigned int>((d_logLuminance[idx] - logLumMin) / logLumRange * numBins));
    atomicAdd(&histo[bin], 1);
}

__global__
void blelloch_scan(unsigned int *g_idata, unsigned int *g_odata, int n)
{
    extern  __shared__  unsigned int temp[];  // allocated on invocation
    
    int thid = threadIdx.x;
    int offset = 1;
    
    temp[2*thid] = g_idata[2*thid];  // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];
    
    for (int d = n>>1; d > 0; d >>= 1)  // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    if (thid == 0) { temp[n - 1] = 0; } // clear the last element
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            unsigned int t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2*thid]   = temp[2*thid]; // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
}
    
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	unsigned int imageSize = numRows * numCols;
	std::cout << "[Debug]imageSize: " << imageSize << std::endl;

    /* 1. find min and max value in d_logLuminance and store them in min_logLum and max_logLum */
	
	// calculate grid and block size for reduce kernel
    unsigned int reduce_gridSize = (imageSize%1024 == 0) ? imageSize/1024 : imageSize/1024+1;  
    unsigned int reduce_blockSize = 1024;
	std::cout << "[Debug]reduce_gridSize: "  << reduce_gridSize  << std::endl;
    std::cout << "[Debug]reduce_blockSize: " << reduce_blockSize << std::endl;
    
    // declare points to max on min value
    float * d_max_logLum, * d_min_logLum;
    
    // allocate memory on device for d_max_logLum and d_min_logLum
    checkCudaErrors(cudaMalloc(&d_max_logLum, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_min_logLum, sizeof(float)));
    
	// call max and min reduce kernel to get max_logLum and min_logLum
	max_reduce<<<reduce_gridSize, reduce_blockSize, sizeof(float)*1024>>>(d_logLuminance, d_max_logLum, imageSize);
    // call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after launching kernel 
    // to make sure that no mistakes were made.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
	min_reduce<<<reduce_gridSize, reduce_blockSize, sizeof(float)*1024>>>(d_logLuminance, d_min_logLum, imageSize);
    // call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after launching kernel 
    // to make sure that no mistakes were made.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
    // copy max and min back to host memory
    checkCudaErrors(cudaMemcpy(&max_logLum, d_max_logLum, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&min_logLum, d_min_logLum, sizeof(float), cudaMemcpyDeviceToHost));
    
    // debug code to see if i got the correct max and min loglum values
    std::cout << "[Debug]max_logLum: " << max_logLum << std::endl;
    std::cout << "[Debug]min_logLum: " << min_logLum << std::endl;
    
	/*2. subtract the minimum value from the maximum value in the input logLuminance channel to get the range */
	float logLumRange = max_logLum - min_logLum;

    // debug code to see if i got the correct max and min loglum values
    std::cout << "[Debug]logLumRange: " << logLumRange << std::endl;
    
	/* 3. generate a histogram of all the values in the logLuminance channel using the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
	
	// declare a point to histogram memory
	unsigned int *histo;
	// allocate memory on device
	checkCudaErrors(cudaMalloc(&histo, sizeof(unsigned int)*numBins));
	// check out cudamemset to initialize histo
	cudaMemset(histo, 0, sizeof(unsigned int)*numBins);
	// calculate grid and block size for histogram kernel
    unsigned int histo_gridSize = (imageSize%1024 == 0) ? imageSize/1024 : imageSize/1024+1; 
    unsigned int histo_blockSize = 1024;
    std::cout << "[Debug]histo_gridSize: "  << histo_gridSize  << std::endl;
    std::cout << "[Debug]histo_blockSize: " << histo_blockSize << std::endl;

    std::cout << "[Debug]numBins: " << numBins << std::endl;
	// launch the histogram kernel to get the histogram of luminance values
    histogram<<<histo_gridSize, histo_blockSize>>>(d_logLuminance, histo, min_logLum, logLumRange, numBins);
    // call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after launching kernel 
    // to make sure that no mistakes were made.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    

	/* 4. Perform an exclusive scan (prefix sum) on the histogram to get the cumulative distribution of luminance values */
	//calculate grid and block size for exclusive scan kernel
	unsigned int scan_gridSize = 1; 
	unsigned int scan_blockSize = numBins/2;
    std::cout << "[Debug]scan_blockSize: " << scan_blockSize << std::endl;
	// launch the Blelloch scan kernel
	blelloch_scan<<<scan_gridSize, scan_blockSize, sizeof(unsigned int)*numBins>>>(histo, d_cdf, numBins);
    // call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after launching kernel 
    // to make sure that no mistakes were made.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 
    
    /* 5. Free allocated device memory */
    cudaFree(d_max_logLum);
    cudaFree(d_min_logLum);
    cudaFree(histo);
    
}

