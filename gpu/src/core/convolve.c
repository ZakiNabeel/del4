/*********************************************************************
 * convolve.c - OpenACC Optimized Version
 * 
 * OpenACC Optimizations Applied (inspired by CUDA version):
 * 1. Persistent device buffers (like CUDA's g_gpu structure)
 * 2. Data stays on device between calls (minimal H2D/D2H transfers)
 * 3. Separable convolution with GPU-to-GPU operations
 * 4. Async operations for overlapping compute/transfer
 * 5. Cache directives for tiling optimization
 * 6. Gang/Vector parallelization for coalesced memory access
 * 7. Optimized gradient computation (single upload, dual output)
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <openacc.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71
#define MAX_KERNEL_SIZE 35

/*********************************************************************
 * Kernel Data Structures
 *********************************************************************/
typedef struct {
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * Persistent Device Buffers (like CUDA g_gpu structure)
 * Keep data on device across multiple calls
 *********************************************************************/
static struct {
  float *d_img1, *d_img2, *d_img_source;  // Device buffers
  size_t allocated_size;
  int initialized;
} g_acc_buffers = {NULL, NULL, NULL, 0, 0};

static void ensure_device_buffers(size_t bytes) {
  if (!g_acc_buffers.initialized) {
    g_acc_buffers.initialized = 1;
  }
  
  if (bytes > g_acc_buffers.allocated_size) {
    // Free old buffers if they exist
    if (g_acc_buffers.d_img1) {
      #pragma acc exit data delete(g_acc_buffers.d_img1[0:g_acc_buffers.allocated_size/sizeof(float)])
      free(g_acc_buffers.d_img1);
    }
    if (g_acc_buffers.d_img2) {
      #pragma acc exit data delete(g_acc_buffers.d_img2[0:g_acc_buffers.allocated_size/sizeof(float)])
      free(g_acc_buffers.d_img2);
    }
    if (g_acc_buffers.d_img_source) {
      #pragma acc exit data delete(g_acc_buffers.d_img_source[0:g_acc_buffers.allocated_size/sizeof(float)])
      free(g_acc_buffers.d_img_source);
    }
    
    // Allocate new buffers
    size_t n_floats = bytes / sizeof(float);
    g_acc_buffers.d_img1 = (float*)malloc(bytes);
    g_acc_buffers.d_img2 = (float*)malloc(bytes);
    g_acc_buffers.d_img_source = (float*)malloc(bytes);
    
    // Create persistent device copies
    #pragma acc enter data create(g_acc_buffers.d_img1[0:n_floats])
    #pragma acc enter data create(g_acc_buffers.d_img2[0:n_floats])
    #pragma acc enter data create(g_acc_buffers.d_img_source[0:n_floats])
    
    g_acc_buffers.allocated_size = bytes;
  }
}

void _KLTCleanupGPU() {
  if (g_acc_buffers.initialized) {
    size_t n_floats = g_acc_buffers.allocated_size / sizeof(float);
    if (g_acc_buffers.d_img1) {
      #pragma acc exit data delete(g_acc_buffers.d_img1[0:n_floats])
      free(g_acc_buffers.d_img1);
    }
    if (g_acc_buffers.d_img2) {
      #pragma acc exit data delete(g_acc_buffers.d_img2[0:n_floats])
      free(g_acc_buffers.d_img2);
    }
    if (g_acc_buffers.d_img_source) {
      #pragma acc exit data delete(g_acc_buffers.d_img_source[0:n_floats])
      free(g_acc_buffers.d_img_source);
    }
    g_acc_buffers.initialized = 0;
    g_acc_buffers.allocated_size = 0;
  }
}

/*********************************************************************
 * _KLTToFloatImage - OpenACC Optimized
 *********************************************************************/
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  int total_size = ncols * nrows;
  float *ptrout = floatimg->data;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  /* Parallel conversion with OpenACC */
  #pragma acc parallel loop independent copyin(img[0:total_size]) copyout(ptrout[0:total_size])
  for (int i = 0; i < total_size; i++) {
    ptrout[i] = (float)img[i];
  }
}

/*********************************************************************
 * _computeKernels - OpenACC with reduction optimizations
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

  /* Compute kernels */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));
    
    /* Compute gauss and deriv - parallel */
    #pragma acc parallel loop
    for (i = -hw; i <= hw; i++) {
      gauss->data[i+hw] = (float)exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths (sequential - data dependent) */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; i++, gaussderiv->width -= 2);
    
    if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for sigma %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  int gauss_w = gauss->width;
  int gaussderiv_w = gaussderiv->width;
  
  #pragma acc parallel loop
  for (i = 0; i < gauss_w; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss_w)/2];
  
  #pragma acc parallel loop
  for (i = 0; i < gaussderiv_w; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv_w)/2];

  /* Normalize gauss and deriv with reduction */
  {
    const int hw = gaussderiv->width / 2;
    float den = 0.0;
    
    #pragma acc parallel loop reduction(+:den)
    for (i = 0; i < gauss->width; i++)
      den += gauss->data[i];
    
    #pragma acc parallel loop
    for (i = 0; i < gauss->width; i++)
      gauss->data[i] /= den;
    
    den = 0.0;
    #pragma acc parallel loop reduction(+:den)
    for (i = -hw; i <= hw; i++)
      den -= i*gaussderiv->data[i+hw];
    
    #pragma acc parallel loop
    for (i = -hw; i <= hw; i++)
      gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
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
 * _convolveImageHoriz - OpenACC Optimized with Tiling
 * 
 * Key optimizations:
 * - Gang/Vector parallelization
 * - Cache directive for kernel data (similar to CUDA constant memory)
 * - Collapse for better GPU utilization
 * - Optimized for coalesced memory access
 *********************************************************************/
static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int radius = kernel.width / 2;
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  int kernel_width = kernel.width;
  
  float *data_in = imgin->data;
  float *data_out = imgout->data;

  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* OpenACC optimized horizontal convolution with cache directive */
  #pragma acc data copyin(data_in[0:ncols*nrows], kernel.data[0:kernel_width]) \
                   copyout(data_out[0:ncols*nrows])
  {
    #pragma acc parallel loop gang vector collapse(2) \
                cache(kernel.data[0:kernel_width])
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int out_idx = j * ncols + i;
        
        if (i < radius || i >= ncols - radius) {
          data_out[out_idx] = 0.0f;
        } else {
          float sum = 0.0f;
          #pragma acc loop seq
          for (int k = 0; k < kernel_width; k++) {
            sum += data_in[j * ncols + (i - radius + k)] * kernel.data[kernel_width - 1 - k];
          }
          data_out[out_idx] = sum;
        }
      }
    }
  }
}

/*********************************************************************
 * _convolveImageVert - OpenACC Optimized
 * 
 * Similar optimizations to horizontal, but with vertical stride
 *********************************************************************/
static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int radius = kernel.width / 2;
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  int kernel_width = kernel.width;
  
  float *data_in = imgin->data;
  float *data_out = imgout->data;

  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  #pragma acc data copyin(data_in[0:ncols*nrows], kernel.data[0:kernel_width]) \
                   copyout(data_out[0:ncols*nrows])
  {
    #pragma acc parallel loop gang vector collapse(2) \
                cache(kernel.data[0:kernel_width])
    for (int j = 0; j < nrows; j++) {
      for (int i = 0; i < ncols; i++) {
        int out_idx = j * ncols + i;
        
        if (j < radius || j >= nrows - radius) {
          data_out[out_idx] = 0.0f;
        } else {
          float sum = 0.0f;
          #pragma acc loop seq
          for (int k = 0; k < kernel_width; k++) {
            sum += data_in[(j - radius + k) * ncols + i] * kernel.data[kernel_width - 1 - k];
          }
          data_out[out_idx] = sum;
        }
      }
    }
  }
}

/*********************************************************************
 * _convolveSeparate - OpenACC Optimized with Persistent Buffers
 * 
 * Key optimization: Keep data on device between horizontal and vertical passes
 * Similar to CUDA version that does GPU→GPU operations
 *********************************************************************/
static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols;
  int nrows = imgin->nrows;
  size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_device_buffers(nbytes);
  
  float *h_in = imgin->data;
  float *h_out = imgout->data;
  float *d_img1 = g_acc_buffers.d_img1;
  float *d_img2 = g_acc_buffers.d_img2;
  
  int h_radius = horiz_kernel.width / 2;
  int v_radius = vert_kernel.width / 2;
  int h_kwidth = horiz_kernel.width;
  int v_kwidth = vert_kernel.width;

  /* Upload input once */
  #pragma acc update device(d_img1[0:ncols*nrows]) async(1)
  #pragma acc host_data use_device(d_img1, d_img2)
  {
    memcpy(d_img1, h_in, nbytes);
  }
  #pragma acc wait(1)
  
  /* Horizontal pass: d_img1 → d_img2 (stays on device) */
  #pragma acc parallel loop gang vector collapse(2) \
              present(d_img1[0:ncols*nrows], d_img2[0:ncols*nrows]) \
              copyin(horiz_kernel.data[0:h_kwidth]) \
              cache(horiz_kernel.data[0:h_kwidth]) async(1)
  for (int j = 0; j < nrows; j++) {
    for (int i = 0; i < ncols; i++) {
      int idx = j * ncols + i;
      if (i < h_radius || i >= ncols - h_radius) {
        d_img2[idx] = 0.0f;
      } else {
        float sum = 0.0f;
        #pragma acc loop seq
        for (int k = 0; k < h_kwidth; k++) {
          sum += d_img1[j * ncols + (i - h_radius + k)] * horiz_kernel.data[h_kwidth - 1 - k];
        }
        d_img2[idx] = sum;
      }
    }
  }
  
  /* Vertical pass: d_img2 → d_img1 (stays on device) */
  #pragma acc parallel loop gang vector collapse(2) \
              present(d_img1[0:ncols*nrows], d_img2[0:ncols*nrows]) \
              copyin(vert_kernel.data[0:v_kwidth]) \
              cache(vert_kernel.data[0:v_kwidth]) async(1)
  for (int j = 0; j < nrows; j++) {
    for (int i = 0; i < ncols; i++) {
      int idx = j * ncols + i;
      if (j < v_radius || j >= nrows - v_radius) {
        d_img1[idx] = 0.0f;
      } else {
        float sum = 0.0f;
        #pragma acc loop seq
        for (int k = 0; k < v_kwidth; k++) {
          sum += d_img2[(j - v_radius + k) * ncols + i] * vert_kernel.data[v_kwidth - 1 - k];
        }
        d_img1[idx] = sum;
      }
    }
  }
  
  /* Download result once */
  #pragma acc update self(d_img1[0:ncols*nrows]) async(1)
  #pragma acc wait(1)
  #pragma acc host_data use_device(d_img1)
  {
    memcpy(h_out, d_img1, nbytes);
  }
  
  imgout->ncols = ncols;
  imgout->nrows = nrows;
}

/*********************************************************************
 * _KLTComputeGradients - OpenACC Optimized with Persistent Buffers
 * 
 * Like CUDA version: Upload once, compute both gradients, download once
 *********************************************************************/
void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  int ncols = img->ncols;
  int nrows = img->nrows;
  size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_device_buffers(nbytes);
  
  float *d_source = g_acc_buffers.d_img_source;
  float *d_img1 = g_acc_buffers.d_img1;
  float *d_img2 = g_acc_buffers.d_img2;
  
  int g_radius = gauss_kernel.width / 2;
  int gd_radius = gaussderiv_kernel.width / 2;
  int g_kwidth = gauss_kernel.width;
  int gd_kwidth = gaussderiv_kernel.width;
  
  /* Upload source image once */
  #pragma acc update device(d_source[0:ncols*nrows])
  memcpy(d_source, img->data, nbytes);
  
  /* ===== COMPUTE GRADX: (gaussderiv_x * gauss_y) ===== */
  
  /* Horizontal gaussderiv pass */
  #pragma acc parallel loop gang vector collapse(2) \
              present(d_source[0:ncols*nrows], d_img1[0:ncols*nrows]) \
              copyin(gaussderiv_kernel.data[0:gd_kwidth]) async(1)
  for (int j = 0; j < nrows; j++) {
    for (int i = 0; i < ncols; i++) {
      int idx = j * ncols + i;
      if (i < gd_radius || i >= ncols - gd_radius) {
        d_img1[idx] = 0.0f;
      } else {
        float sum = 0.0f;
        for (int k = 0; k < gd_kwidth; k++) {
          sum += d_source[j * ncols + (i - gd_radius + k)] * gaussderiv_kernel.data[gd_kwidth - 1 - k];
        }
        d_img1[idx] = sum;
      }
    }
  }
  
  /* Vertical gauss pass → gradx output */
  #pragma acc parallel loop gang vector collapse(2) \
              present(d_img1[0:ncols*nrows], d_img2[0:ncols*nrows]) \
              copyin(gauss_kernel.data[0:g_kwidth]) async(1)
  for (int j = 0; j < nrows; j++) {
    for (int i = 0; i < ncols; i++) {
      int idx = j * ncols + i;
      if (j < g_radius || j >= nrows - g_radius) {
        d_img2[idx] = 0.0f;
      } else {
        float sum = 0.0f;
        for (int k = 0; k < g_kwidth; k++) {
          sum += d_img1[(j - g_radius + k) * ncols + i] * gauss_kernel.data[g_kwidth - 1 - k];
        }
        d_img2[idx] = sum;
      }
    }
  }
  
  /* Download gradx */
  #pragma acc update self(d_img2[0:ncols*nrows]) async(1)
  #pragma acc wait(1)
  memcpy(gradx->data, d_img2, nbytes);
  
  /* ===== COMPUTE GRADY: (gauss_x * gaussderiv_y) ===== */
  
  /* Horizontal gauss pass */
  #pragma acc parallel loop gang vector collapse(2) \
              present(d_source[0:ncols*nrows], d_img1[0:ncols*nrows]) \
              copyin(gauss_kernel.data[0:g_kwidth]) async(2)
  for (int j = 0; j < nrows; j++) {
    for (int i = 0; i < ncols; i++) {
      int idx = j * ncols + i;
      if (i < g_radius || i >= ncols - g_radius) {
        d_img1[idx] = 0.0f;
      } else {
        float sum = 0.0f;
        for (int k = 0; k < g_kwidth; k++) {
          sum += d_source[j * ncols + (i - g_radius + k)] * gauss_kernel.data[g_kwidth - 1 - k];
        }
        d_img1[idx] = sum;
      }
    }
  }
  
  /* Vertical gaussderiv pass → grady output */
  #pragma acc parallel loop gang vector collapse(2) \
              present(d_img1[0:ncols*nrows], d_source[0:ncols*nrows]) \
              copyin(gaussderiv_kernel.data[0:gd_kwidth]) async(2)
  for (int j = 0; j < nrows; j++) {
    for (int i = 0; i < ncols; i++) {
      int idx = j * ncols + i;
      if (j < gd_radius || j >= nrows - gd_radius) {
        d_source[idx] = 0.0f;
      } else {
        float sum = 0.0f;
        for (int k = 0; k < gd_kwidth; k++) {
          sum += d_img1[(j - gd_radius + k) * ncols + i] * gaussderiv_kernel.data[gd_kwidth - 1 - k];
        }
        d_source[idx] = sum;
      }
    }
  }
  
  /* Download grady */
  #pragma acc update self(d_source[0:ncols*nrows]) async(2)
  #pragma acc wait(2)
  memcpy(grady->data, d_source, nbytes);
  
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
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}
