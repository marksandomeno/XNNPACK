// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

static const int64_t mask_table[7] = {-1, -1, -1, -1, 0, 0, 0};

void xnn_x64_transposec_ukernel__4x4_reuse_multi_avx(
    const uint64_t* input,
    uint64_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint64_t));
  assert(input_stride >= block_width * sizeof(uint64_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_hbytes = tile_height * sizeof(uint64_t);
  const size_t tile_wbytes = tile_width * sizeof(uint64_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint64_t);

  const uint64_t* i0 = input;
  uint64_t* o0 = (uint64_t*) output;
  uint64_t* o1 = (uint64_t*) ((uintptr_t) o0 + output_stride);
  uint64_t* o2 = (uint64_t*) ((uintptr_t) o1 + output_stride);
  uint64_t* o3 = (uint64_t*) ((uintptr_t) o2 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    const size_t rem = min(block_width - 1, 3);

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[3 - rem]));

    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      const __m256i v2_0 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint64_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v2_1 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint64_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v2_2 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint64_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v2_3 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint64_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v1_0 = _mm256_unpacklo_pd(v2_0, v2_1);
      const __m256i v1_1 = _mm256_unpackhi_pd(v2_0, v2_1);
      const __m256i v1_2 = _mm256_unpacklo_pd(v2_2, v2_3);
      const __m256i v1_3 = _mm256_unpackhi_pd(v2_2, v2_3);

      const __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_2, 0x20);
      const __m256i v0_2 = _mm256_permute2f128_ps(v1_0, v1_2, 0x31);
      const __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_3, 0x20);
      const __m256i v0_3 = _mm256_permute2f128_ps(v1_1, v1_3, 0x31);

      _mm256_storeu_si256((__m256i*) o3, v0_3);
      o3 = (uint64_t*) ((uintptr_t) o3 + tile_hbytes);
      _mm256_storeu_si256((__m256i*) o2, v0_2);
      o2 = (uint64_t*) ((uintptr_t) o2 + tile_hbytes);
      _mm256_storeu_si256((__m256i*) o1, v0_1);
      o1 = (uint64_t*) ((uintptr_t) o1 + tile_hbytes);
      _mm256_storeu_si256((__m256i*) o0, v0_0);
      o0 = (uint64_t*) ((uintptr_t) o0 + tile_hbytes);
    }
    if (bh != 0) {
      const __m256i v2_0 = _mm256_maskload_ps((const float*) i0, vmask);
      const uint64_t *i1 = (const uint64_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v2_1 = _mm256_maskload_ps((const float*) i1, vmask);
      const uint64_t *i2 = (const uint64_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v2_2 = _mm256_maskload_ps((const float*) i2, vmask);
      const __m256i v2_3 = _mm256_undefined_si256();

      const __m256i v1_0 = _mm256_unpacklo_pd(v2_0, v2_1);
      const __m256i v1_1 = _mm256_unpackhi_pd(v2_0, v2_1);
      const __m256i v1_2 = _mm256_unpacklo_pd(v2_2, v2_3);
      const __m256i v1_3 = _mm256_unpackhi_pd(v2_2, v2_3);

      __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_2, 0x20);
      __m256i v0_2 = _mm256_permute2f128_ps(v1_0, v1_2, 0x31);
      __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_3, 0x20);
      __m256i v0_3 = _mm256_permute2f128_ps(v1_1, v1_3, 0x31);


      if (bh & 2) {
        _mm_storeu_si128((__m128i*) o3, _mm256_castsi256_si128(v0_3));
        o3 += 2;
        _mm_storeu_si128((__m128i*) o2, _mm256_castsi256_si128(v0_2));
        o2 += 2;
        _mm_storeu_si128((__m128i*) o1, _mm256_castsi256_si128(v0_1));
        o1 += 2;
        _mm_storeu_si128((__m128i*) o0, _mm256_castsi256_si128(v0_0));
        o0 += 2;
        v0_0 = _mm256_permute2f128_ps(v0_0, v0_0, 0x1);
        v0_1 = _mm256_permute2f128_ps(v0_1, v0_1, 0x1);
        v0_2 = _mm256_permute2f128_ps(v0_2, v0_2, 0x1);
        v0_3 = _mm256_permute2f128_ps(v0_3, v0_3, 0x1);
      }

      if (bh & 1) {
        _mm_storel_epi64((__m128i*) o3, _mm256_castsi256_si128(v0_3));
        _mm_storel_epi64((__m128i*) o2, _mm256_castsi256_si128(v0_2));
        _mm_storel_epi64((__m128i*) o1, _mm256_castsi256_si128(v0_1));
        _mm_storel_epi64((__m128i*) o0, _mm256_castsi256_si128(v0_0));
      }
    }

    i0 = (const uint64_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint64_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint64_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint64_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint64_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
