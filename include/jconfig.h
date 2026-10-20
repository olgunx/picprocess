#ifndef JPEG_LIB_VERSION
#if defined(__arm__) || defined(__aarch64__)
/* Debian / Raspberry Pi OS standard libjpeg is libjpeg.so.62 (API version 62) */
#define JPEG_LIB_VERSION  62
#else
/* Ubuntu / x86_64 standard libjpeg is libjpeg.so.8 (API version 80) */
#define JPEG_LIB_VERSION  80
#endif
#endif

/* libjpeg-turbo version */
#define LIBJPEG_TURBO_VERSION  2.1.5

/* libjpeg-turbo version in integer form */
#define LIBJPEG_TURBO_VERSION_NUMBER  2001005

/* Support arithmetic encoding */
#define C_ARITH_CODING_SUPPORTED 1

/* Support arithmetic decoding */
#define D_ARITH_CODING_SUPPORTED 1

/* Support in-memory source/destination managers */
/* #undef MEM_SRCDST_SUPPORTED */

/* Use accelerated SIMD routines. */
#define WITH_SIMD 1

/*
 * Define BITS_IN_JSAMPLE as either
 *   8   for 8-bit sample values (the usual setting)
 *   12  for 12-bit sample values
 * Only 8 and 12 are legal data precisions for lossy JPEG according to the
 * JPEG standard, and the IJG code does not support anything else!
 * We do not support run-time selection of data precision, sorry.
 */

#define BITS_IN_JSAMPLE  8      /* use 8 or 12 */

/* Define if your (broken) compiler shifts signed values as if they were
   unsigned. */
/* #undef RIGHT_SHIFT_IS_UNSIGNED */
