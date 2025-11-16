/*********************************************************************
 * selectGoodFeatures.c - OpenACC optimized version
 * 
 * OpenACC Optimizations (matching CUDA selectGoodFeatures_cuda.cu):
 * 1. Eigenvalue computation on GPU (parallel loop)
 * 2. Bitonic sort on GPU (replace CPU quicksort)
 * 3. Tail padding for sort on GPU
 * 4. Min-distance enforcement on CPU (simple & robust)
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <openacc.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"

int KLT_verbose = 0;

typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

/*********************************************************************
 * Utility: next power of 2 (for bitonic sort)
 *********************************************************************/
static inline unsigned int next_pow2(unsigned int x) {
  if (x <= 1) return 1;
  x--; 
  x |= x >> 1; 
  x |= x >> 2; 
  x |= x >> 4; 
  x |= x >> 8; 
  x |= x >> 16;
  return x + 1;
}

/*********************************************************************
 * GPU Eigenvalue Computation (OpenACC)
 * Computes min eigenvalue for each candidate (x,y) and stores [x,y,val]
 *********************************************************************/
static void compute_eigenvalues_gpu(
  float *gradx_data, float *grady_data,
  int ncols, int nrows,
  int window_hw, int window_hh,
  int borderx, int bordery,
  int skip,
  int *pointlist, int *npoints_out)
{
  int stride = skip + 1;
  int x_count = ((ncols - 2 * borderx) - 1) / stride + 1;
  int y_count = ((nrows - 2 * bordery) - 1) / stride + 1;
  int npoints = x_count * y_count;
  
  *npoints_out = npoints;
  
  // GPU eigenvalue computation
  #pragma acc data copyin(gradx_data[0:ncols*nrows], grady_data[0:ncols*nrows]) \
                   copyout(pointlist[0:npoints*3])
  {
    #pragma acc parallel loop gang worker vector_length(256) collapse(2)
    for (int yi = 0; yi < y_count; yi++) {
      for (int xi = 0; xi < x_count; xi++) {
        int x = borderx + xi * stride;
        int y = bordery + yi * stride;
        
        float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
        
        // Sum gradients in window
        #pragma acc loop seq
        for (int yy = y - window_hh; yy <= y + window_hh; yy++) {
          #pragma acc loop seq
          for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            int idx = yy * ncols + xx;
            float gx = gradx_data[idx];
            float gy = grady_data[idx];
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
          }
        }
        
        // Compute minimum eigenvalue
        float trace = gxx + gyy;
        float det = gxx * gyy - gxy * gxy;
        float disc = (trace * trace - 4.0f * det);
        if (disc < 0.0f) disc = 0.0f;
        float minev = 0.5f * (trace - sqrtf(disc));
        
        // Clamp to INT_MAX-1
        int val = (int)(minev < (float)(INT_MAX - 1) ? minev : (float)(INT_MAX - 1));
        
        // Store triplet [x, y, val]
        int out_idx = yi * x_count + xi;
        pointlist[out_idx * 3 + 0] = x;
        pointlist[out_idx * 3 + 1] = y;
        pointlist[out_idx * 3 + 2] = val;
      }
    }
  }
}

/*********************************************************************
 * GPU Tail Padding (OpenACC)
 * Pad tail triplets with sentinel {0, 0, INT_MIN} for bitonic sort
 *********************************************************************/
static void pad_tail_gpu(int *pointlist, int start, int count)
{
  #pragma acc parallel loop gang vector present(pointlist[0:(start+count)*3])
  for (int i = 0; i < count; i++) {
    int idx = (start + i) * 3;
    pointlist[idx + 0] = 0;
    pointlist[idx + 1] = 0;
    pointlist[idx + 2] = INT_MIN;
  }
}

/*********************************************************************
 * GPU Bitonic Sort (OpenACC)
 * In-place bitonic sort by val descending
 *********************************************************************/
static void bitonic_sort_gpu(int *pointlist, int n)
{
  if (n <= 1) return;
  
  unsigned int m = next_pow2((unsigned int)n);
  
  // Pad tail if needed
  if (m > (unsigned int)n) {
    int pad_count = (int)(m - (unsigned int)n);
    
    #pragma acc data present(pointlist[0:m*3])
    {
      #pragma acc parallel loop gang vector
      for (int i = 0; i < pad_count; i++) {
        int idx = (n + i) * 3;
        pointlist[idx + 0] = 0;
        pointlist[idx + 1] = 0;
        pointlist[idx + 2] = INT_MIN;
      }
    }
  }
  
  // Bitonic sort
  #pragma acc data present(pointlist[0:m*3])
  {
    for (unsigned int k = 2; k <= m; k <<= 1) {
      for (unsigned int j = k >> 1; j > 0; j >>= 1) {
        
        #pragma acc parallel loop gang vector
        for (unsigned int i = 0; i < m; i++) {
          unsigned int ixj = i ^ j;
          if (ixj <= i || ixj >= m) continue;
          
          int vi = pointlist[3 * i + 2];
          int vj = pointlist[3 * ixj + 2];
          
          int ascending_half = ((i & k) == 0);
          int swap = ascending_half ? (vi < vj) : (vi > vj);
          
          if (swap) {
            // Swap triplets
            int tx = pointlist[3 * i + 0];
            int ty = pointlist[3 * i + 1];
            int tv = pointlist[3 * i + 2];
            
            pointlist[3 * i + 0] = pointlist[3 * ixj + 0];
            pointlist[3 * i + 1] = pointlist[3 * ixj + 1];
            pointlist[3 * i + 2] = pointlist[3 * ixj + 2];
            
            pointlist[3 * ixj + 0] = tx;
            pointlist[3 * ixj + 1] = ty;
            pointlist[3 * ixj + 2] = tv;
          }
        }
        
        #pragma acc wait
      }
    }
  }
}

/*********************************************************************
 * CPU: Fill featuremap (unchanged from original)
 *********************************************************************/
static void _fillFeaturemap(
  int x, int y, 
  uchar *featuremap, 
  int mindist, 
  int ncols, 
  int nrows)
{
  for (int iy = y - mindist ; iy <= y + mindist ; iy++)
    for (int ix = x - mindist ; ix <= x + mindist ; ix++)
      if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
        featuremap[iy*ncols+ix] = 1;
}

/*********************************************************************
 * CPU: Enforce minimum distance (unchanged from original, robust)
 *********************************************************************/
static void _enforceMinimumDistance(
  int *pointlist,
  int npoints,
  KLT_FeatureList featurelist,
  int ncols, int nrows,
  int mindist,
  int min_eigenvalue,
  KLT_BOOL overwriteAllFeatures)
{
  int indx;
  int x, y, val;
  uchar *featuremap;
  int *ptr;
	
  if (min_eigenvalue < 1)  min_eigenvalue = 1;

  featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
  memset(featuremap, 0, ncols*nrows);
	
  mindist--;

  if (!overwriteAllFeatures)
    for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
      if (featurelist->feature[indx]->val >= 0)  {
        x   = (int) featurelist->feature[indx]->x;
        y   = (int) featurelist->feature[indx]->y;
        _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
      }

  ptr = pointlist;
  indx = 0;
  while (1)  {
    if (ptr >= pointlist + 3*npoints)  {
      while (indx < featurelist->nFeatures)  {	
        if (overwriteAllFeatures || 
            featurelist->feature[indx]->val < 0) {
          featurelist->feature[indx]->x   = -1;
          featurelist->feature[indx]->y   = -1;
          featurelist->feature[indx]->val = KLT_NOT_FOUND;
          featurelist->feature[indx]->aff_img = NULL;
          featurelist->feature[indx]->aff_img_gradx = NULL;
          featurelist->feature[indx]->aff_img_grady = NULL;
          featurelist->feature[indx]->aff_x = -1.0;
          featurelist->feature[indx]->aff_y = -1.0;
          featurelist->feature[indx]->aff_Axx = 1.0;
          featurelist->feature[indx]->aff_Ayx = 0.0;
          featurelist->feature[indx]->aff_Axy = 0.0;
          featurelist->feature[indx]->aff_Ayy = 1.0;
        }
        indx++;
      }
      break;
    }

    x   = *ptr++;
    y   = *ptr++;
    val = *ptr++;
		
    assert(x >= 0);
    assert(x < ncols);
    assert(y >= 0);
    assert(y < nrows);
	
    while (!overwriteAllFeatures && 
           indx < featurelist->nFeatures &&
           featurelist->feature[indx]->val >= 0)
      indx++;

    if (indx >= featurelist->nFeatures)  break;

    if (!featuremap[y*ncols+x] && val >= min_eigenvalue)  {
      featurelist->feature[indx]->x   = (KLT_locType) x;
      featurelist->feature[indx]->y   = (KLT_locType) y;
      featurelist->feature[indx]->val = (int) val;
      featurelist->feature[indx]->aff_img = NULL;
      featurelist->feature[indx]->aff_img_gradx = NULL;
      featurelist->feature[indx]->aff_img_grady = NULL;
      featurelist->feature[indx]->aff_x = -1.0;
      featurelist->feature[indx]->aff_y = -1.0;
      featurelist->feature[indx]->aff_Axx = 1.0;
      featurelist->feature[indx]->aff_Ayx = 0.0;
      featurelist->feature[indx]->aff_Axy = 0.0;
      featurelist->feature[indx]->aff_Ayy = 1.0;
      indx++;

      _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
    }
  }

  free(featuremap);
}

/*********************************************************************
 * Main selection function with OpenACC GPU acceleration
 *********************************************************************/
void _KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList featurelist,
  selectionMode mode)
{
  _KLT_FloatImage floatimg, gradx, grady;
  int window_hw, window_hh;
  int *pointlist;
  int npoints = 0;
  KLT_BOOL overwriteAllFeatures = (mode == SELECTING_ALL) ? TRUE : FALSE;
  KLT_BOOL floatimages_created = FALSE;

  /* Check window size */
  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd. Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd. Changing to %d.\n", tc->window_height);
  }
  if (tc->window_width < 3) {
    tc->window_width = 3;
    KLTWarning("Tracking context's window width must be at least three. Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height < 3) {
    tc->window_height = 3;
    KLTWarning("Tracking context's window height must be at least three. Changing to %d.\n", tc->window_height);
  }
  window_hw = tc->window_width/2; 
  window_hh = tc->window_height/2;
		
  /* Compute max possible points */
  int borderx = tc->borderx;
  int bordery = tc->bordery;
  if (borderx < window_hw)  borderx = window_hw;
  if (bordery < window_hh)  bordery = window_hh;
  
  int stride = tc->nSkippedPixels + 1;
  int x_count = ((ncols - 2 * borderx) > 0) ? ((ncols - 2 * borderx - 1) / stride + 1) : 0;
  int y_count = ((nrows - 2 * bordery) > 0) ? ((nrows - 2 * bordery - 1) / stride + 1) : 0;
  int max_points = x_count * y_count;
  unsigned int m = next_pow2((unsigned int)max_points);
  
  /* Allocate pointlist for padded size */
  pointlist = (int *) malloc(m * 3 * sizeof(int));

  /* Create temporary images */
  if (mode == REPLACING_SOME && 
      tc->sequentialMode && tc->pyramid_last != NULL)  {
    floatimg = ((_KLT_Pyramid) tc->pyramid_last)->img[0];
    gradx = ((_KLT_Pyramid) tc->pyramid_last_gradx)->img[0];
    grady = ((_KLT_Pyramid) tc->pyramid_last_grady)->img[0];
    assert(gradx != NULL);
    assert(grady != NULL);
  } else  {
    floatimages_created = TRUE;
    floatimg = _KLTCreateFloatImage(ncols, nrows);
    gradx    = _KLTCreateFloatImage(ncols, nrows);
    grady    = _KLTCreateFloatImage(ncols, nrows);
    if (tc->smoothBeforeSelecting)  {
      _KLT_FloatImage tmpimg;
      tmpimg = _KLTCreateFloatImage(ncols, nrows);
      _KLTToFloatImage(img, ncols, nrows, tmpimg);
      _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
      _KLTFreeFloatImage(tmpimg);
    } else _KLTToFloatImage(img, ncols, nrows, floatimg);
 
    /* Compute gradients */
    _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
  }
	
  /* Write internal images */
  if (tc->writeInternalImages)  {
    _KLTWriteFloatImageToPGM(floatimg, "kltimg_sgfrlf.pgm");
    _KLTWriteFloatImageToPGM(gradx, "kltimg_sgfrlf_gx.pgm");
    _KLTWriteFloatImageToPGM(grady, "kltimg_sgfrlf_gy.pgm");
  }

  /* GPU: Compute eigenvalues for all candidate points */
  compute_eigenvalues_gpu(
    gradx->data, grady->data,
    ncols, nrows,
    window_hw, window_hh,
    borderx, bordery,
    tc->nSkippedPixels,
    pointlist, &npoints);

  /* GPU: Sort features by eigenvalue descending */
  if (npoints > 0) {
    #pragma acc enter data create(pointlist[0:m*3])
    #pragma acc update device(pointlist[0:npoints*3])
    
    bitonic_sort_gpu(pointlist, npoints);
    
    #pragma acc update self(pointlist[0:npoints*3])
    #pragma acc exit data delete(pointlist[0:m*3])
  }

  /* CPU: Enforce minimum distance */
  if (tc->mindist < 0)  {
    KLTWarning("(_KLTSelectGoodFeatures) Tracking context field tc->mindist is negative (%d); setting to zero", tc->mindist);
    tc->mindist = 0;
  }

  _enforceMinimumDistance(
    pointlist,
    npoints,
    featurelist,
    ncols, nrows,
    tc->mindist,
    tc->min_eigenvalue,
    overwriteAllFeatures);

  /* Free memory */
  free(pointlist);
  if (floatimages_created)  {
    _KLTFreeFloatImage(floatimg);
    _KLTFreeFloatImage(gradx);
    _KLTFreeFloatImage(grady);
  }
}

/*********************************************************************
 * Public API functions
 *********************************************************************/
void KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows,
  KLT_FeatureList featurelist)
{
  if (KLT_verbose >= 1) {
    fprintf(stderr, "(KLT) Selecting the %d best features from a %d by %d image...  ", 
            featurelist->nFeatures, ncols, nrows);
    fflush(stderr);
  }

  _KLTSelectGoodFeatures(tc, img, ncols, nrows, featurelist, SELECTING_ALL);

  if (KLT_verbose >= 1) {
    fprintf(stderr, "\n\t%d features found.\n", 
            KLTCountRemainingFeatures(featurelist));
    if (tc->writeInternalImages)
      fprintf(stderr, "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

void KLTReplaceLostFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows,
  KLT_FeatureList featurelist)
{
  int nLost = featurelist->nFeatures - KLTCountRemainingFeatures(featurelist);

  if (KLT_verbose >= 1) {
    fprintf(stderr, "(KLT) Attempting to replace %d features in a %d by %d image...  ", 
            nLost, ncols, nrows);
    fflush(stderr);
  }

  if (nLost > 0) {
    _KLTSelectGoodFeatures(tc, img, ncols, nrows, featurelist, REPLACING_SOME);
  }

  if (KLT_verbose >= 1) {
    fprintf(stderr, "\n\t%d features replaced.\n",
            nLost - featurelist->nFeatures + KLTCountRemainingFeatures(featurelist));
    if (tc->writeInternalImages)
      fprintf(stderr, "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}
