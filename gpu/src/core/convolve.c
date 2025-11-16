/*********************************************************************
 * convolve.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */

#define MAX_KERNEL_WIDTH 	71


typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  int npix = ncols * nrows;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  // Run conversion on GPU
  #pragma acc parallel loop copyin(img[0:npix]) copyout(floatimg->data[0:npix])
  for (int i = 0; i < npix; ++i) {
    floatimg->data[i] = (float) img[i];
  }
}

/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

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
 * _convolveImageHoriz
 */

static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrrow = imgin->data;
  float *ptrout = imgout->data;

  int ncols  = imgin->ncols;
  int nrows  = imgin->nrows;
  int radius = kernel.width / 2;
  int j;

  assert(kernel.width % 2 == 1);

  /* Parallelise over rows; each thread processes one row */
  #pragma acc parallel loop present(imgin->data[0:ncols*nrows], \
                                    imgout->data[0:ncols*nrows], \
                                    kernel.data[0:kernel.width])
  for (j = 0 ; j < nrows ; j++) {

    float *row_in  = ptrrow  + j * ncols;
    float *row_out = ptrout  + j * ncols;
    int i, k;

    /* Zero leftmost columns */
    for (i = 0 ; i < radius ; i++)
      row_out[i] = 0.0f;

    /* Convolve middle columns */
    for ( ; i < ncols - radius ; i++) {
      float sum = 0.0f;
      const float *ppp = row_in + i - radius;
      for (k = 0; k < kernel.width; ++k)
        sum += ppp[k] * kernel.data[k];
      row_out[i] = sum;
    }

    /* Zero rightmost columns */
    for ( ; i < ncols ; i++)
      row_out[i] = 0.0f;
  }
}

/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrcol = imgin->data;
  float *ptrout = imgout->data;

  int ncols  = imgin->ncols;
  int nrows  = imgin->nrows;
  int radius = kernel.width / 2;
  int i;

  assert(kernel.width % 2 == 1);

  #pragma acc parallel loop present(imgin->data[0:ncols*nrows], \
                                    imgout->data[0:ncols*nrows], \
                                    kernel.data[0:kernel.width])
  for (i = 0 ; i < ncols ; i++) {
    int j, k;

    /* Zero top rows */
    for (j = 0 ; j < radius ; j++)
      ptrout[j*ncols + i] = 0.0f;

    /* Convolve middle rows */
    for ( ; j < nrows - radius ; j++) {
      float sum = 0.0f;
      const float *ppp = ptrcol + (j - radius)*ncols + i;
      for (k = 0 ; k < kernel.width ; ++k) {
        sum += ppp[k*ncols] * kernel.data[k];
      }
      ptrout[j*ncols + i] = sum;
    }

    /* Zero bottom rows */
    for ( ; j < nrows ; j++)
      ptrout[j*ncols + i] = 0.0f;
  }
}

/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  int fullN = imgin->ncols * imgin->nrows;

  #pragma acc data \
    copyin(imgin->data[0:fullN], \
           horiz_kernel.data[0:horiz_kernel.width], \
           vert_kernel.data[0:vert_kernel.width]) \
    create(tmpimg->data[0:fullN]) \
    copyout(imgout->data[0:fullN])
  {
    _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
    _convolveImageVert(tmpimg, vert_kernel, imgout);
  }

  _KLTFreeFloatImage(tmpimg);
}

/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);

}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

