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

void xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2(
    const uint16_t* input,
    uint16_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint16_t));
  assert(input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t) - tile_hbytes;

  const uint16_t* i0 = input;
  uint16_t* o = (uint16_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 15);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7 - (rem>>1)]));

    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const __m256i v4_0 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_1 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_2 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_3 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_4 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_5 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_6 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_7 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_8 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_9 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_10 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_11 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_12 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_13 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_14 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_15 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v3_0 = _mm256_unpacklo_epi16(v4_0, v4_4);
      const __m256i v3_1 = _mm256_unpackhi_epi16(v4_0, v4_4);
      const __m256i v3_2 = _mm256_unpacklo_epi16(v4_1, v4_5);
      const __m256i v3_3 = _mm256_unpackhi_epi16(v4_1, v4_5);
      const __m256i v3_4 = _mm256_unpacklo_epi16(v4_2, v4_6);
      const __m256i v3_5 = _mm256_unpackhi_epi16(v4_2, v4_6);
      const __m256i v3_6 = _mm256_unpacklo_epi16(v4_3, v4_7);
      const __m256i v3_7 = _mm256_unpackhi_epi16(v4_3, v4_7);
      const __m256i v3_8 = _mm256_unpacklo_epi16(v4_8, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi16(v4_8, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi16(v4_9, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi16(v4_9, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi16(v4_10, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi16(v4_10, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi16(v4_11, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi16(v4_11, v4_15);
      const __m256i v2_0 = _mm256_unpacklo_epi16(v3_0, v3_4);
      const __m256i v2_1 = _mm256_unpackhi_epi16(v3_0, v3_4);
      const __m256i v2_2 = _mm256_unpacklo_epi16(v3_1, v3_5);
      const __m256i v2_3 = _mm256_unpackhi_epi16(v3_1, v3_5);
      const __m256i v2_4 = _mm256_unpacklo_epi16(v3_2, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_epi16(v3_2, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_epi16(v3_3, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_epi16(v3_3, v3_7);
      const __m256i v2_8 = _mm256_unpacklo_epi16(v3_8, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi16(v3_8, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi16(v3_9, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi16(v3_9, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi16(v3_10, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi16(v3_10, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi16(v3_11, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi16(v3_11, v3_15);
      const __m256i v1_0 = _mm256_unpacklo_epi16(v2_0, v2_4);
      const __m256i v1_1 = _mm256_unpackhi_epi16(v2_0, v2_4);
      const __m256i v1_2 = _mm256_unpacklo_epi16(v2_1, v2_5);
      const __m256i v1_3 = _mm256_unpackhi_epi16(v2_1, v2_5);
      const __m256i v1_4 = _mm256_unpacklo_epi16(v2_2, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_epi16(v2_2, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_epi16(v2_3, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_epi16(v2_3, v2_7);
      const __m256i v1_8 = _mm256_unpacklo_epi16(v2_8, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi16(v2_8, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi16(v2_9, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi16(v2_9, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi16(v2_10, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi16(v2_10, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi16(v2_11, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi16(v2_11, v2_15);

      const __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_8, 0x20);
      const __m256i v0_8 = _mm256_permute2f128_ps(v1_0, v1_8, 0x31);
      const __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_9, 0x20);
      const __m256i v0_9 = _mm256_permute2f128_ps(v1_1, v1_9, 0x31);
      const __m256i v0_2 = _mm256_permute2f128_ps(v1_2, v1_10, 0x20);
      const __m256i v0_10 = _mm256_permute2f128_ps(v1_2, v1_10, 0x31);
      const __m256i v0_3 = _mm256_permute2f128_ps(v1_3, v1_11, 0x20);
      const __m256i v0_11 = _mm256_permute2f128_ps(v1_3, v1_11, 0x31);
      const __m256i v0_4 = _mm256_permute2f128_ps(v1_4, v1_12, 0x20);
      const __m256i v0_12 = _mm256_permute2f128_ps(v1_4, v1_12, 0x31);
      const __m256i v0_5 = _mm256_permute2f128_ps(v1_5, v1_13, 0x20);
      const __m256i v0_13 = _mm256_permute2f128_ps(v1_5, v1_13, 0x31);
      const __m256i v0_6 = _mm256_permute2f128_ps(v1_6, v1_14, 0x20);
      const __m256i v0_14 = _mm256_permute2f128_ps(v1_6, v1_14, 0x31);
      const __m256i v0_7 = _mm256_permute2f128_ps(v1_7, v1_15, 0x20);
      const __m256i v0_15 = _mm256_permute2f128_ps(v1_7, v1_15, 0x31);

      o = (uint16_t*) ((uintptr_t) o + oN_offset);
      _mm256_storeu_si256((__m256i*) o, v0_15);
      uint16_t *oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 15) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_14);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 15) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_13);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 13) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_12);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 13) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_11);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 11) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_10);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 11) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_9);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 9) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_8);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 9) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_7);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 7) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_6);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 7) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_5);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 5) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_4);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 5) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_3);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 3) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_2);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 3) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_1);
      oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_0);
    }
    o = (uint16_t*) ((uintptr_t) o + tile_hbytes);
    if (bh != 0) {
      const __m256i v4_0 = _mm256_maskload_ps((const float*) i0, vmask);
      const uint16_t *i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v4_1 = _mm256_maskload_ps((const float*) i1, vmask);
      const uint16_t *i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v4_2 = _mm256_maskload_ps((const float*) i2, vmask);
      const uint16_t *i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256i v4_3 = _mm256_maskload_ps((const float*) i3, vmask);
      const uint16_t *i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256i v4_4 = _mm256_maskload_ps((const float*) i4, vmask);
      const uint16_t *i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256i v4_5 = _mm256_maskload_ps((const float*) i5, vmask);
      const uint16_t *i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256i v4_6 = _mm256_maskload_ps((const float*) i6, vmask);
      const uint16_t *i7 = (const uint16_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const __m256i v4_7 = _mm256_maskload_ps((const float*) i7, vmask);
      const uint16_t *i8 = (const uint16_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const __m256i v4_8 = _mm256_maskload_ps((const float*) i8, vmask);
      const uint16_t *i9 = (const uint16_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const __m256i v4_9 = _mm256_maskload_ps((const float*) i9, vmask);
      const uint16_t *i10 = (const uint16_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const __m256i v4_10 = _mm256_maskload_ps((const float*) i10, vmask);
      const uint16_t *i11 = (const uint16_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const __m256i v4_11 = _mm256_maskload_ps((const float*) i11, vmask);
      const uint16_t *i12 = (const uint16_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const __m256i v4_12 = _mm256_maskload_ps((const float*) i12, vmask);
      const uint16_t *i13 = (const uint16_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const __m256i v4_13 = _mm256_maskload_ps((const float*) i13, vmask);
      const uint16_t *i14 = (const uint16_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const __m256i v4_14 = _mm256_maskload_ps((const float*) i14, vmask);
      const __m256i v4_15 = _mm256_undefined_si256();

      const __m256i v3_0 = _mm256_unpacklo_epi16(v4_0, v4_4);
      const __m256i v3_1 = _mm256_unpackhi_epi16(v4_0, v4_4);
      const __m256i v3_2 = _mm256_unpacklo_epi16(v4_1, v4_5);
      const __m256i v3_3 = _mm256_unpackhi_epi16(v4_1, v4_5);
      const __m256i v3_4 = _mm256_unpacklo_epi16(v4_2, v4_6);
      const __m256i v3_5 = _mm256_unpackhi_epi16(v4_2, v4_6);
      const __m256i v3_6 = _mm256_unpacklo_epi16(v4_3, v4_7);
      const __m256i v3_7 = _mm256_unpackhi_epi16(v4_3, v4_7);
      const __m256i v3_8 = _mm256_unpacklo_epi16(v4_8, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi16(v4_8, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi16(v4_9, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi16(v4_9, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi16(v4_10, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi16(v4_10, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi16(v4_11, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi16(v4_11, v4_15);
      const __m256i v2_0 = _mm256_unpacklo_epi16(v3_0, v3_4);
      const __m256i v2_1 = _mm256_unpackhi_epi16(v3_0, v3_4);
      const __m256i v2_2 = _mm256_unpacklo_epi16(v3_1, v3_5);
      const __m256i v2_3 = _mm256_unpackhi_epi16(v3_1, v3_5);
      const __m256i v2_4 = _mm256_unpacklo_epi16(v3_2, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_epi16(v3_2, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_epi16(v3_3, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_epi16(v3_3, v3_7);
      const __m256i v2_8 = _mm256_unpacklo_epi16(v3_8, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi16(v3_8, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi16(v3_9, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi16(v3_9, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi16(v3_10, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi16(v3_10, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi16(v3_11, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi16(v3_11, v3_15);
      const __m256i v1_0 = _mm256_unpacklo_epi16(v2_0, v2_4);
      const __m256i v1_1 = _mm256_unpackhi_epi16(v2_0, v2_4);
      const __m256i v1_2 = _mm256_unpacklo_epi16(v2_1, v2_5);
      const __m256i v1_3 = _mm256_unpackhi_epi16(v2_1, v2_5);
      const __m256i v1_4 = _mm256_unpacklo_epi16(v2_2, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_epi16(v2_2, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_epi16(v2_3, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_epi16(v2_3, v2_7);
      const __m256i v1_8 = _mm256_unpacklo_epi16(v2_8, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi16(v2_8, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi16(v2_9, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi16(v2_9, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi16(v2_10, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi16(v2_10, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi16(v2_11, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi16(v2_11, v2_15);

      __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_8, 0x20);
      __m256i v0_8 = _mm256_permute2f128_ps(v1_0, v1_8, 0x31);
      __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_9, 0x20);
      __m256i v0_9 = _mm256_permute2f128_ps(v1_1, v1_9, 0x31);
      __m256i v0_2 = _mm256_permute2f128_ps(v1_2, v1_10, 0x20);
      __m256i v0_10 = _mm256_permute2f128_ps(v1_2, v1_10, 0x31);
      __m256i v0_3 = _mm256_permute2f128_ps(v1_3, v1_11, 0x20);
      __m256i v0_11 = _mm256_permute2f128_ps(v1_3, v1_11, 0x31);
      __m256i v0_4 = _mm256_permute2f128_ps(v1_4, v1_12, 0x20);
      __m256i v0_12 = _mm256_permute2f128_ps(v1_4, v1_12, 0x31);
      __m256i v0_5 = _mm256_permute2f128_ps(v1_5, v1_13, 0x20);
      __m256i v0_13 = _mm256_permute2f128_ps(v1_5, v1_13, 0x31);
      __m256i v0_6 = _mm256_permute2f128_ps(v1_6, v1_14, 0x20);
      __m256i v0_14 = _mm256_permute2f128_ps(v1_6, v1_14, 0x31);
      __m256i v0_7 = _mm256_permute2f128_ps(v1_7, v1_15, 0x20);
      __m256i v0_15 = _mm256_permute2f128_ps(v1_7, v1_15, 0x31);


      if (bh & 8) {
        o = (uint16_t*) ((uintptr_t) o + oN_stride);
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_15));
        uint16_t *oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_14));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_13));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_12));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_11));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_10));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_9));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_8));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_7));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_6));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_5));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_4));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_3));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_2));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_1));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_0));
        o += 8;
        v0_0 = _mm256_permute2f128_ps(v0_0, v0_0, 0x1);
        v0_1 = _mm256_permute2f128_ps(v0_1, v0_1, 0x1);
        v0_2 = _mm256_permute2f128_ps(v0_2, v0_2, 0x1);
        v0_3 = _mm256_permute2f128_ps(v0_3, v0_3, 0x1);
        v0_4 = _mm256_permute2f128_ps(v0_4, v0_4, 0x1);
        v0_5 = _mm256_permute2f128_ps(v0_5, v0_5, 0x1);
        v0_6 = _mm256_permute2f128_ps(v0_6, v0_6, 0x1);
        v0_7 = _mm256_permute2f128_ps(v0_7, v0_7, 0x1);
        v0_8 = _mm256_permute2f128_ps(v0_8, v0_8, 0x1);
        v0_9 = _mm256_permute2f128_ps(v0_9, v0_9, 0x1);
        v0_10 = _mm256_permute2f128_ps(v0_10, v0_10, 0x1);
        v0_11 = _mm256_permute2f128_ps(v0_11, v0_11, 0x1);
        v0_12 = _mm256_permute2f128_ps(v0_12, v0_12, 0x1);
        v0_13 = _mm256_permute2f128_ps(v0_13, v0_13, 0x1);
        v0_14 = _mm256_permute2f128_ps(v0_14, v0_14, 0x1);
        v0_15 = _mm256_permute2f128_ps(v0_15, v0_15, 0x1);
      }

      if (bh & 4) {
        o = (uint16_t*) ((uintptr_t) o + oN_stride);
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_15));
        uint16_t *oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_14));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_13));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_12));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_11));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_10));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_9));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_8));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_7));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_6));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_5));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_4));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_3));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_2));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_1));
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_0));
        o += 4;
        v0_0 = _mm256_unpackhi_pd(v0_0, v0_0);
        v0_1 = _mm256_unpackhi_pd(v0_1, v0_1);
        v0_2 = _mm256_unpackhi_pd(v0_2, v0_2);
        v0_3 = _mm256_unpackhi_pd(v0_3, v0_3);
        v0_4 = _mm256_unpackhi_pd(v0_4, v0_4);
        v0_5 = _mm256_unpackhi_pd(v0_5, v0_5);
        v0_6 = _mm256_unpackhi_pd(v0_6, v0_6);
        v0_7 = _mm256_unpackhi_pd(v0_7, v0_7);
        v0_8 = _mm256_unpackhi_pd(v0_8, v0_8);
        v0_9 = _mm256_unpackhi_pd(v0_9, v0_9);
        v0_10 = _mm256_unpackhi_pd(v0_10, v0_10);
        v0_11 = _mm256_unpackhi_pd(v0_11, v0_11);
        v0_12 = _mm256_unpackhi_pd(v0_12, v0_12);
        v0_13 = _mm256_unpackhi_pd(v0_13, v0_13);
        v0_14 = _mm256_unpackhi_pd(v0_14, v0_14);
        v0_15 = _mm256_unpackhi_pd(v0_15, v0_15);
      }
      if (bh & 2) {
        o = (uint16_t*) ((uintptr_t) o + oN_stride);
        *((int*) o) = _mm256_cvtsi256_si32(v0_15);
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_14);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_13);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_12);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_11);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_10);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_9);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_8);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_7);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_6);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_5);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_4);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_3);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_2);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_1);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_0);
        o += 2;
        v0_0 = _mm256_srli_epi64(v0_0, 32);
        v0_1 = _mm256_srli_epi64(v0_1, 32);
        v0_2 = _mm256_srli_epi64(v0_2, 32);
        v0_3 = _mm256_srli_epi64(v0_3, 32);
        v0_4 = _mm256_srli_epi64(v0_4, 32);
        v0_5 = _mm256_srli_epi64(v0_5, 32);
        v0_6 = _mm256_srli_epi64(v0_6, 32);
        v0_7 = _mm256_srli_epi64(v0_7, 32);
        v0_8 = _mm256_srli_epi64(v0_8, 32);
        v0_9 = _mm256_srli_epi64(v0_9, 32);
        v0_10 = _mm256_srli_epi64(v0_10, 32);
        v0_11 = _mm256_srli_epi64(v0_11, 32);
        v0_12 = _mm256_srli_epi64(v0_12, 32);
        v0_13 = _mm256_srli_epi64(v0_13, 32);
        v0_14 = _mm256_srli_epi64(v0_14, 32);
        v0_15 = _mm256_srli_epi64(v0_15, 32);
      }
      if (bh & 1) {
        o = (uint16_t*) ((uintptr_t) o + oN_stride);
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_15);
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_14);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_13);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_12);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_11);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_10);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_9);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_8);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_7);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_6);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_5);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_4);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_3);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_2);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_1);
        oN = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_0);
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    o = (uint16_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
