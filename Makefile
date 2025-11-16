/*********************************************************************
 * convolve.c - PURE OpenACC version (no CUDA patterns, no double copies)
 * 
 * OpenACC Optimizations:
 * 1. Single data regions for fused separable convolution
 * 2. create() for temporary buffers (stay on GPU between passes)
 * 3. Gang-worker-vector parallelism with collapse(2)
 * 4. Kernel reversal (matches CPU convolution behavior)
 * 5. Direct copyin/copyout (no double copies via memcpy)
 * 6. Vector length tuning (256 threads per gang)
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <openacc.h>

/* Local includes */
#include "base.h"
#include "error.h"
#include "convolve.h"

#define MAX_KERNEL_WIDTH 71

typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Static kernels */
static ConvolutionKernel gauss_kernel, gaussderiv_kernel;

/*********************************************************************
 * _computeKernels - CPU only
 *********************************************************************/
static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));
	
    for (i = -hw; i <= hw; i++) {
      gauss->data[i+hw] = (float)exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH) {
      fprintf(stderr, "ERROR: MAX_KERNEL_WIDTH %d is too small for sigma of %f\n",
              MAX_KERNEL_WIDTH, sigma);
      exit(1);
    }
  }

  for (i = 0; i < gauss->width; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0; i < gaussderiv->width; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];

  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0; i < gauss->width; i++) den += gauss->data[i];
    for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw; i <= hw; i++) den -= i*gaussderiv->data[i+hw];
    for (i = -hw; i <= hw; i++) gaussderiv->data[i+hw] /= den;
  }
}

/*********************************************************************
 * _KLTToFloatImage - OpenACC optimized
 *********************************************************************/
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  int npix = ncols * nrows;
  float *ptrfl = floatimg->data;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  #pragma acc parallel loop gang vector vector_length(256) \
          copyin(img[0:npix]) copyout(ptrfl[0:npix])
  for (int i = 0; i < npix; i++) {
    ptrfl[i] = (float)img[i];
  }
}

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * _convolveSeparate - PURE OpenACC separable convolution
 * Single data region: copyin input, create tmp (stays on GPU), copyout result
 *********************************************************************/
static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  int npix = ncols * nrows;
  
  float *in_data = imgin->data;
  float *out_data = imgout->data;
  
  // Allocate temporary buffer on host
  float *tmp_data = (float*)malloc(npix * sizeof(float));
  
  // Reverse kernels (match CUDA logic)
  int h_kwidth = horiz_kernel.width;
  int h_radius = h_kwidth / 2;
  float h_kernel_rev[MAX_KERNEL_WIDTH];
  for (int k = 0; k < h_kwidth; k++) {
    h_kernel_rev[k] = horiz_kernel.data[h_kwidth - 1 - k];
  }
  
  int v_kwidth = vert_kernel.width;
  int v_radius = v_kwidth / 2;
  float v_kernel_rev[MAX_KERNEL_WIDTH];
  for (int k = 0; k < v_kwidth; k++) {
    v_kernel_rev[k] = vert_kernel.data[v_kwidth - 1 - k];
  }
  
  // ============ SINGLE DATA REGION FOR ENTIRE SEPARABLE CONVOLUTION ============
  // copyin: input image + kernels
  // create: temporary buffer (stays on GPU between passes)
  // copyout: final output image
  #pragma acc data copyin(in_data[0:npix], h_kernel_rev[0:h_kwidth], v_kernel_rev[0:v_kwidth]) \
                   create(tmp_data[0:npix]) \
                   copyout(out_data[0:npix])
  {
    // ============ HORIZONTAL PASS: in_data -> tmp_data ============
    #pragma acc parallel loop gang worker vector_length(256) collapse(2)
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int idx = j * ncols + i;
        
        if (i < h_radius || i >= ncols - h_radius) {
          tmp_data[idx] = 0.0f;
        } else {
          float sum = 0.0f;
          int base = j * ncols + i - h_radius;
          
          #pragma acc loop seq
          for (int k = 0; k < h_kwidth; k++) {
            sum += in_data[base + k] * h_kernel_rev[k];
          }
          tmp_data[idx] = sum;
        }
      }
    }
    
    // ============ VERTICAL PASS: tmp_data -> out_data ============
    #pragma acc parallel loop gang worker vector_length(256) collapse(2)
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int idx = j * ncols + i;
        
        if (j < v_radius || j >= nrows - v_radius) {
          out_data[idx] = 0.0f;
        } else {
          float sum = 0.0f;
          int base = (j - v_radius) * ncols + i;
          
          #pragma acc loop seq
          for (int k = 0; k < v_kwidth; k++) {
            sum += tmp_data[base + k * ncols] * v_kernel_rev[k];
          }
          out_data[idx] = sum;
        }
      }
    }
  } // Data region ends - automatic copyout of out_data
  
  free(tmp_data);
  
  imgout->ncols = ncols;
  imgout->nrows = nrows;
}
	
/*********************************************************************
 * _KLTComputeGradients - PURE OpenACC (no double copies)
 *********************************************************************/
void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  /* Compute kernels */
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  int ncols = img->ncols;
  int nrows = img->nrows;
  int npix = ncols * nrows;
  
  float *img_data = img->data;
  float *gradx_data = gradx->data;
  float *grady_data = grady->data;
  
  // Allocate temporary buffers
  float *tmp1 = (float*)malloc(npix * sizeof(float));
  float *tmp2 = (float*)malloc(npix * sizeof(float));
  
  // Reverse kernels
  int gd_kwidth = gaussderiv_kernel.width;
  int gd_radius = gd_kwidth / 2;
  int g_kwidth = gauss_kernel.width;
  int g_radius = g_kwidth / 2;
  
  float gauss_rev[MAX_KERNEL_WIDTH], gaussderiv_rev[MAX_KERNEL_WIDTH];
  for (int k = 0; k < g_kwidth; k++) {
    gauss_rev[k] = gauss_kernel.data[g_kwidth - 1 - k];
  }
  for (int k = 0; k < gd_kwidth; k++) {
    gaussderiv_rev[k] = gaussderiv_kernel.data[gd_kwidth - 1 - k];
  }
  
  // ============ COMPUTE GRADX: (gaussderiv_x * gauss_y) ============
  #pragma acc data copyin(img_data[0:npix], gauss_rev[0:g_kwidth], gaussderiv_rev[0:gd_kwidth]) \
                   create(tmp1[0:npix], tmp2[0:npix]) \
                   copyout(gradx_data[0:npix])
  {
    // Horizontal pass with gaussderiv
    #pragma acc parallel loop gang worker vector_length(256) collapse(2)
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int idx = j * ncols + i;
        if (i < gd_radius || i >= ncols - gd_radius) {
          tmp1[idx] = 0.0f;
        } else {
          float sum = 0.0f;
          int base = j * ncols + i - gd_radius;
          #pragma acc loop seq
          for (int k = 0; k < gd_kwidth; k++) {
            sum += img_data[base + k] * gaussderiv_rev[k];
          }
          tmp1[idx] = sum;
        }
      }
    }
    
    // Vertical pass with gauss
    #pragma acc parallel loop gang worker vector_length(256) collapse(2)
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int idx = j * ncols + i;
        if (j < g_radius || j >= nrows - g_radius) {
          gradx_data[idx] = 0.0f;
        } else {
          float sum = 0.0f;
          int base = (j - g_radius) * ncols + i;
          #pragma acc loop seq
          for (int k = 0; k < g_kwidth; k++) {
            sum += tmp1[base + k * ncols] * gauss_rev[k];
          }
          gradx_data[idx] = sum;
        }
      }
    }
  }
  
  // ============ COMPUTE GRADY: (gauss_x * gaussderiv_y) ============
  #pragma acc data copyin(img_data[0:npix], gauss_rev[0:g_kwidth], gaussderiv_rev[0:gd_kwidth]) \
                   create(tmp1[0:npix], tmp2[0:npix]) \
                   copyout(grady_data[0:npix])
  {
    // Horizontal pass with gauss
    #pragma acc parallel loop gang worker vector_length(256) collapse(2)
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int idx = j * ncols + i;
        if (i < g_radius || i >= ncols - g_radius) {
          tmp1[idx] = 0.0f;
        } else {
          float sum = 0.0f;
          int base = j * ncols + i - g_radius;
          #pragma acc loop seq
          for (int k = 0; k < g_kwidth; k++) {
            sum += img_data[base + k] * gauss_rev[k];
          }
          tmp1[idx] = sum;
        }
      }
    }
    
    // Vertical pass with gaussderiv
    #pragma acc parallel loop gang worker vector_length(256) collapse(2)
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int idx = j * ncols + i;
        if (j < gd_radius || j >= nrows - gd_radius) {
          grady_data[idx] = 0.0f;
        } else {
          float sum = 0.0f;
          int base = (j - gd_radius) * ncols + i;
          #pragma acc loop seq
          for (int k = 0; k < gd_kwidth; k++) {
            sum += tmp1[base + k * ncols] * gaussderiv_rev[k];
          }
          grady_data[idx] = sum;
        }
      }
    }
  }
  
  free(tmp1);
  free(tmp2);
  
  gradx->ncols = ncols;
  gradx->nrows = nrows;
  grady->ncols = ncols;
  grady->nrows = nrows;
}

/*********************************************************************
 * _KLTComputeSmoothedImage
 *********************************************************************/
void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Compute kernel */
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  /* Do convolution */
  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

/*********************************************************************
 * Cleanup function - no persistent buffers in pure OpenACC version
 *********************************************************************/
void _KLTCleanupGPU() {
  // Pure OpenACC version manages memory automatically via data regions
  // No manual cleanup needed
}
