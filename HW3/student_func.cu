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
	
  /* 1. find min and max value in d_logLuminance and store them in min_logLum and max_logLum */
	
	// calculate grid and block size for reduce kernel
  unsigned int reduce_gridSize = (imageSize%1024 == 0) ? imageSize/1024 : imageSize/1024+1;  
  unsigned int reduce_blockSize = 1024;
	
	// call max and min reduce kernel to get max_logLum and min_logLum
	max_reduce<<<gridSize, blockSize, sizeof(float)*1024>>>(d_logLuminance, max_logLum);
	min_reduce<<<gridSize, blockSize, sizeof(float)*1024>>>(d_logLuminance, min_logLum);
	
	/*2. subtract the minimum value from the maximum value in the input logLuminance channel to get the range */
	float logLumRange = logLumMax - logLumMin;
	
	/* 3. generate a histogram of all the values in the logLuminance channel using the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
	
	// declare a point to histogram memory
	unsigned int *histo;
	// allocate memory on device
	checkCudaErrors(cudaMalloc(&histo, sizeof(unsigned int)*numBins);
	// check out cudamemset to initialize histo
	cudaMemset(histo, 0x00, sizeof(unsigned int)*numBins);
	// calculate grid and block size for histogram kernel
  unsigned int histo_gridSize = (imageSize%1024 == 0) ? imageSize/1024 : imageSize/1024+1; 
  unsigned int histo_blockSize = 1024;
	// launch the histogram kernel to get the histogram of luminance values
  histogram<<<histo_gridSize, histo_blockSize>>>(d_ logLuminance, histo, logLumMin, logLumRange);
	
	/* 4. Perform an exclusive scan (prefix sum) on the histogram to get the cumulative distribution of luminance values */
	//calculate grid and block size for exclusive scan kernel
	unsigned int scan_gridSize = 	(numBins%1024 == 0) ? numBins/1024 : numBins/1024+1;  
	unsigned int scan_blockSize = 1024;
	// launch the scan kernel
	scan<<<scan_gridSize, scan_blockSize>>>(histo, d_cdf, numBins);
}

__global__ 	
void max_reduce(const float* const d_logLuminance, 
								float &max_logLum)
{
	// smax and smin are allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	// How do i allocate two pieces of shared memory?
  extern __shared__ float sdata[];
		
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

	// load shared mem from global mem
  sdata[tid] = d_logLuminance[idx];
	__syncthreads(); 			// make sure entire block is loaded!

  // do max and min reduction in global mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
      if (tid < s)
      {
          sdata[tid] = max(d_logLuminance[tid + s], sdata[tid]);
      }
      __syncthreads(); // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
  	max_logLum = max(sdata[tid], max_logLum);
  }

}

__global__ 	
void min_reduce(const float* const d_logLuminance, 
												float &min_logLum)
{
	// smax and smin are allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	// How do i allocate two pieces of shared memory?
  extern __shared__ float sdata[];
	
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

	// load shared mem from global mem
  sdata[tid] = d_logLuminance[idx];
	__syncthreads(); 			// make sure entire block is loaded!

  // do max and min reduction in global mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
      if (tid < s)
      {
  			sdata[tid] = min(d_logLuminance[tid + s], sdata[tid]);
      }
      __syncthreads(); // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
		min_logLum = min(sdata[tid], min_logLum);
  }

}

__global__
void histogram(const float* const d_logLuminance, 
							unsigned int* histo,
							float &logLumMin,
							float &logLumRange)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                    static_cast<unsigned int>((d_logLuminance[idx] - logLumMin) / logLumRange * numBins));
  histo[bin]++;
}

__global__
void scan(unsigned int* histo, 
					unsigned int* const d_cdf,
					const size_t numBins)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx == 0)
		d_cdf[idx] = 0;
	d_cdf[idx] = d_cdf[idx-1] + histo[idx-1];
}
