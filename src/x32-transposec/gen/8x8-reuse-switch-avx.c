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

static const int32_t mask_table[15] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_x32_transposec_ukernel__8x8_reuse_switch_avx(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint32_t));
  assert(input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 8;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;
  uint32_t* o = (uint32_t*) output;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 7);
    const size_t oN_stride = rem * output_stride;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7 - rem]));

    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const __m256i v3_0 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v3_1 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v3_2 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v3_3 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v3_4 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v3_5 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v3_6 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v3_7 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v2_0 = _mm256_unpacklo_ps(v3_0, v3_2);
      const __m256i v2_1 = _mm256_unpackhi_ps(v3_0, v3_2);
      const __m256i v2_2 = _mm256_unpacklo_ps(v3_1, v3_3);
      const __m256i v2_3 = _mm256_unpackhi_ps(v3_1, v3_3);
      const __m256i v2_4 = _mm256_unpacklo_ps(v3_4, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_ps(v3_4, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_ps(v3_5, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_ps(v3_5, v3_7);
      const __m256i v1_0 = _mm256_unpacklo_ps(v2_0, v2_2);
      const __m256i v1_1 = _mm256_unpackhi_ps(v2_0, v2_2);
      const __m256i v1_2 = _mm256_unpacklo_ps(v2_1, v2_3);
      const __m256i v1_3 = _mm256_unpackhi_ps(v2_1, v2_3);
      const __m256i v1_4 = _mm256_unpacklo_ps(v2_4, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_ps(v2_4, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_ps(v2_5, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_ps(v2_5, v2_7);

      const __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_4, 0x20);
      const __m256i v0_4 = _mm256_permute2f128_ps(v1_0, v1_4, 0x31);
      const __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_5, 0x20);
      const __m256i v0_5 = _mm256_permute2f128_ps(v1_1, v1_5, 0x31);
      const __m256i v0_2 = _mm256_permute2f128_ps(v1_2, v1_6, 0x20);
      const __m256i v0_6 = _mm256_permute2f128_ps(v1_2, v1_6, 0x31);
      const __m256i v0_3 = _mm256_permute2f128_ps(v1_3, v1_7, 0x20);
      const __m256i v0_7 = _mm256_permute2f128_ps(v1_3, v1_7, 0x31);

      uint32_t* oN = (uint32_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        case 7:
          _mm256_storeu_si256((__m256i*) oN, v0_7);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
        case 6:
          _mm256_storeu_si256((__m256i*) oN, v0_6);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
        case 5:
          _mm256_storeu_si256((__m256i*) oN, v0_5);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
        case 4:
          _mm256_storeu_si256((__m256i*) oN, v0_4);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
        case 3:
          _mm256_storeu_si256((__m256i*) oN, v0_3);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
        case 2:
          _mm256_storeu_si256((__m256i*) oN, v0_2);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
        case 1:
          _mm256_storeu_si256((__m256i*) oN, v0_1);
        case 0:
          _mm256_storeu_si256((__m256i*) o, v0_0);
          o = (uint32_t*) ((uintptr_t) o + tile_hbytes);
          break;
        default:
          XNN_UNREACHABLE;
      }
    }
    if (bh != 0) {
      const __m256i v3_0 = _mm256_maskload_ps((const float*) i0, vmask);
      const uint32_t *i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v3_1 = _mm256_maskload_ps((const float*) i1, vmask);
      const uint32_t *i2 = (const uint32_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v3_2 = _mm256_maskload_ps((const float*) i2, vmask);
      const uint32_t *i3 = (const uint32_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256i v3_3 = _mm256_maskload_ps((const float*) i3, vmask);
      const uint32_t *i4 = (const uint32_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256i v3_4 = _mm256_maskload_ps((const float*) i4, vmask);
      const uint32_t *i5 = (const uint32_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256i v3_5 = _mm256_maskload_ps((const float*) i5, vmask);
      const uint32_t *i6 = (const uint32_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256i v3_6 = _mm256_maskload_ps((const float*) i6, vmask);
      const __m256i v3_7 = _mm256_undefined_si256();

      const __m256i v2_0 = _mm256_unpacklo_ps(v3_0, v3_2);
      const __m256i v2_1 = _mm256_unpackhi_ps(v3_0, v3_2);
      const __m256i v2_2 = _mm256_unpacklo_ps(v3_1, v3_3);
      const __m256i v2_3 = _mm256_unpackhi_ps(v3_1, v3_3);
      const __m256i v2_4 = _mm256_unpacklo_ps(v3_4, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_ps(v3_4, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_ps(v3_5, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_ps(v3_5, v3_7);
      const __m256i v1_0 = _mm256_unpacklo_ps(v2_0, v2_2);
      const __m256i v1_1 = _mm256_unpackhi_ps(v2_0, v2_2);
      const __m256i v1_2 = _mm256_unpacklo_ps(v2_1, v2_3);
      const __m256i v1_3 = _mm256_unpackhi_ps(v2_1, v2_3);
      const __m256i v1_4 = _mm256_unpacklo_ps(v2_4, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_ps(v2_4, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_ps(v2_5, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_ps(v2_5, v2_7);

      __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_4, 0x20);
      __m256i v0_4 = _mm256_permute2f128_ps(v1_0, v1_4, 0x31);
      __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_5, 0x20);
      __m256i v0_5 = _mm256_permute2f128_ps(v1_1, v1_5, 0x31);
      __m256i v0_2 = _mm256_permute2f128_ps(v1_2, v1_6, 0x20);
      __m256i v0_6 = _mm256_permute2f128_ps(v1_2, v1_6, 0x31);
      __m256i v0_3 = _mm256_permute2f128_ps(v1_3, v1_7, 0x20);
      __m256i v0_7 = _mm256_permute2f128_ps(v1_3, v1_7, 0x31);


      if (bh & 4) {
        uint32_t* oN = (uint32_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            _mm_storeu_si128((__m128i*) oN, _mm256_castsi256_si128(v0_7));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storeu_si128((__m128i*) oN, _mm256_castsi256_si128(v0_6));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storeu_si128((__m128i*) oN, _mm256_castsi256_si128(v0_5));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storeu_si128((__m128i*) oN, _mm256_castsi256_si128(v0_4));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storeu_si128((__m128i*) oN, _mm256_castsi256_si128(v0_3));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storeu_si128((__m128i*) oN, _mm256_castsi256_si128(v0_2));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storeu_si128((__m128i*) oN, _mm256_castsi256_si128(v0_1));
          case 0:
            _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_0));
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 4;
        v0_0 = _mm256_permute2f128_ps(v0_0, v0_0, 0x1);
        v0_1 = _mm256_permute2f128_ps(v0_1, v0_1, 0x1);
        v0_2 = _mm256_permute2f128_ps(v0_2, v0_2, 0x1);
        v0_3 = _mm256_permute2f128_ps(v0_3, v0_3, 0x1);
        v0_4 = _mm256_permute2f128_ps(v0_4, v0_4, 0x1);
        v0_5 = _mm256_permute2f128_ps(v0_5, v0_5, 0x1);
        v0_6 = _mm256_permute2f128_ps(v0_6, v0_6, 0x1);
        v0_7 = _mm256_permute2f128_ps(v0_7, v0_7, 0x1);
      }

      if (bh & 2) {
        uint32_t* oN = (uint32_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            _mm_storel_epi64((__m128i*) oN, _mm256_castsi256_si128(v0_7));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            _mm_storel_epi64((__m128i*) oN, _mm256_castsi256_si128(v0_6));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            _mm_storel_epi64((__m128i*) oN, _mm256_castsi256_si128(v0_5));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            _mm_storel_epi64((__m128i*) oN, _mm256_castsi256_si128(v0_4));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            _mm_storel_epi64((__m128i*) oN, _mm256_castsi256_si128(v0_3));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            _mm_storel_epi64((__m128i*) oN, _mm256_castsi256_si128(v0_2));
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            _mm_storel_epi64((__m128i*) oN, _mm256_castsi256_si128(v0_1));
          case 0:
            _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_0));
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 2;
        v0_0 = _mm256_unpackhi_pd(v0_0, v0_0);
        v0_1 = _mm256_unpackhi_pd(v0_1, v0_1);
        v0_2 = _mm256_unpackhi_pd(v0_2, v0_2);
        v0_3 = _mm256_unpackhi_pd(v0_3, v0_3);
        v0_4 = _mm256_unpackhi_pd(v0_4, v0_4);
        v0_5 = _mm256_unpackhi_pd(v0_5, v0_5);
        v0_6 = _mm256_unpackhi_pd(v0_6, v0_6);
        v0_7 = _mm256_unpackhi_pd(v0_7, v0_7);
      }
      if (bh & 1) {
        uint32_t* oN = (uint32_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            *((int*) oN) = _mm256_cvtsi256_si32(v0_7);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            *((int*) oN) = _mm256_cvtsi256_si32(v0_6);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            *((int*) oN) = _mm256_cvtsi256_si32(v0_5);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            *((int*) oN) = _mm256_cvtsi256_si32(v0_4);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            *((int*) oN) = _mm256_cvtsi256_si32(v0_3);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            *((int*) oN) = _mm256_cvtsi256_si32(v0_2);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            *((int*) oN) = _mm256_cvtsi256_si32(v0_1);
          case 0:
            *((int*) o) = _mm256_cvtsi256_si32(v0_0);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);
    o = (uint32_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
