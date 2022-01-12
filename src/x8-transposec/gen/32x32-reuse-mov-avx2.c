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

void xnn_x8_transposec_ukernel__32x32_reuse_mov_avx2(
    const uint8_t* input,
    uint8_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint8_t));
  assert(input_stride >= block_width * sizeof(uint8_t));

  const size_t tile_height = 32;
  const size_t tile_width = 32;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t) - tile_hbytes;

  const uint8_t* i0 = input;
  uint8_t* o = (uint8_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 31);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7 - (rem>>2)]));

    size_t bh = block_height;
    for (; bh >= 32; bh -= 32) {
      const __m256i v5_0 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_1 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_2 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_3 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_4 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_5 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_6 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_7 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_8 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_9 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_10 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_11 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_12 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_13 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_14 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_15 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_16 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_17 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_18 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_19 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_20 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_21 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_22 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_23 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_24 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_25 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_26 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_27 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_28 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_29 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_30 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_31 = _mm256_maskload_ps((const float*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v4_0 = _mm256_unpacklo_epi8(v5_0, v5_8);
      const __m256i v4_1 = _mm256_unpackhi_epi8(v5_0, v5_8);
      const __m256i v4_2 = _mm256_unpacklo_epi8(v5_1, v5_9);
      const __m256i v4_3 = _mm256_unpackhi_epi8(v5_1, v5_9);
      const __m256i v4_4 = _mm256_unpacklo_epi8(v5_2, v5_10);
      const __m256i v4_5 = _mm256_unpackhi_epi8(v5_2, v5_10);
      const __m256i v4_6 = _mm256_unpacklo_epi8(v5_3, v5_11);
      const __m256i v4_7 = _mm256_unpackhi_epi8(v5_3, v5_11);
      const __m256i v4_8 = _mm256_unpacklo_epi8(v5_4, v5_12);
      const __m256i v4_9 = _mm256_unpackhi_epi8(v5_4, v5_12);
      const __m256i v4_10 = _mm256_unpacklo_epi8(v5_5, v5_13);
      const __m256i v4_11 = _mm256_unpackhi_epi8(v5_5, v5_13);
      const __m256i v4_12 = _mm256_unpacklo_epi8(v5_6, v5_14);
      const __m256i v4_13 = _mm256_unpackhi_epi8(v5_6, v5_14);
      const __m256i v4_14 = _mm256_unpacklo_epi8(v5_7, v5_15);
      const __m256i v4_15 = _mm256_unpackhi_epi8(v5_7, v5_15);
      const __m256i v4_16 = _mm256_unpacklo_epi8(v5_16, v5_24);
      const __m256i v4_17 = _mm256_unpackhi_epi8(v5_16, v5_24);
      const __m256i v4_18 = _mm256_unpacklo_epi8(v5_17, v5_25);
      const __m256i v4_19 = _mm256_unpackhi_epi8(v5_17, v5_25);
      const __m256i v4_20 = _mm256_unpacklo_epi8(v5_18, v5_26);
      const __m256i v4_21 = _mm256_unpackhi_epi8(v5_18, v5_26);
      const __m256i v4_22 = _mm256_unpacklo_epi8(v5_19, v5_27);
      const __m256i v4_23 = _mm256_unpackhi_epi8(v5_19, v5_27);
      const __m256i v4_24 = _mm256_unpacklo_epi8(v5_20, v5_28);
      const __m256i v4_25 = _mm256_unpackhi_epi8(v5_20, v5_28);
      const __m256i v4_26 = _mm256_unpacklo_epi8(v5_21, v5_29);
      const __m256i v4_27 = _mm256_unpackhi_epi8(v5_21, v5_29);
      const __m256i v4_28 = _mm256_unpacklo_epi8(v5_22, v5_30);
      const __m256i v4_29 = _mm256_unpackhi_epi8(v5_22, v5_30);
      const __m256i v4_30 = _mm256_unpacklo_epi8(v5_23, v5_31);
      const __m256i v4_31 = _mm256_unpackhi_epi8(v5_23, v5_31);
      const __m256i v3_0 = _mm256_unpacklo_epi8(v4_0, v4_8);
      const __m256i v3_1 = _mm256_unpackhi_epi8(v4_0, v4_8);
      const __m256i v3_2 = _mm256_unpacklo_epi8(v4_1, v4_9);
      const __m256i v3_3 = _mm256_unpackhi_epi8(v4_1, v4_9);
      const __m256i v3_4 = _mm256_unpacklo_epi8(v4_2, v4_10);
      const __m256i v3_5 = _mm256_unpackhi_epi8(v4_2, v4_10);
      const __m256i v3_6 = _mm256_unpacklo_epi8(v4_3, v4_11);
      const __m256i v3_7 = _mm256_unpackhi_epi8(v4_3, v4_11);
      const __m256i v3_8 = _mm256_unpacklo_epi8(v4_4, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi8(v4_4, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi8(v4_5, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi8(v4_5, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi8(v4_6, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi8(v4_6, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi8(v4_7, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi8(v4_7, v4_15);
      const __m256i v3_16 = _mm256_unpacklo_epi8(v4_16, v4_24);
      const __m256i v3_17 = _mm256_unpackhi_epi8(v4_16, v4_24);
      const __m256i v3_18 = _mm256_unpacklo_epi8(v4_17, v4_25);
      const __m256i v3_19 = _mm256_unpackhi_epi8(v4_17, v4_25);
      const __m256i v3_20 = _mm256_unpacklo_epi8(v4_18, v4_26);
      const __m256i v3_21 = _mm256_unpackhi_epi8(v4_18, v4_26);
      const __m256i v3_22 = _mm256_unpacklo_epi8(v4_19, v4_27);
      const __m256i v3_23 = _mm256_unpackhi_epi8(v4_19, v4_27);
      const __m256i v3_24 = _mm256_unpacklo_epi8(v4_20, v4_28);
      const __m256i v3_25 = _mm256_unpackhi_epi8(v4_20, v4_28);
      const __m256i v3_26 = _mm256_unpacklo_epi8(v4_21, v4_29);
      const __m256i v3_27 = _mm256_unpackhi_epi8(v4_21, v4_29);
      const __m256i v3_28 = _mm256_unpacklo_epi8(v4_22, v4_30);
      const __m256i v3_29 = _mm256_unpackhi_epi8(v4_22, v4_30);
      const __m256i v3_30 = _mm256_unpacklo_epi8(v4_23, v4_31);
      const __m256i v3_31 = _mm256_unpackhi_epi8(v4_23, v4_31);
      const __m256i v2_0 = _mm256_unpacklo_epi8(v3_0, v3_8);
      const __m256i v2_1 = _mm256_unpackhi_epi8(v3_0, v3_8);
      const __m256i v2_2 = _mm256_unpacklo_epi8(v3_1, v3_9);
      const __m256i v2_3 = _mm256_unpackhi_epi8(v3_1, v3_9);
      const __m256i v2_4 = _mm256_unpacklo_epi8(v3_2, v3_10);
      const __m256i v2_5 = _mm256_unpackhi_epi8(v3_2, v3_10);
      const __m256i v2_6 = _mm256_unpacklo_epi8(v3_3, v3_11);
      const __m256i v2_7 = _mm256_unpackhi_epi8(v3_3, v3_11);
      const __m256i v2_8 = _mm256_unpacklo_epi8(v3_4, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi8(v3_4, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi8(v3_5, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi8(v3_5, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi8(v3_6, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi8(v3_6, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi8(v3_7, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi8(v3_7, v3_15);
      const __m256i v2_16 = _mm256_unpacklo_epi8(v3_16, v3_24);
      const __m256i v2_17 = _mm256_unpackhi_epi8(v3_16, v3_24);
      const __m256i v2_18 = _mm256_unpacklo_epi8(v3_17, v3_25);
      const __m256i v2_19 = _mm256_unpackhi_epi8(v3_17, v3_25);
      const __m256i v2_20 = _mm256_unpacklo_epi8(v3_18, v3_26);
      const __m256i v2_21 = _mm256_unpackhi_epi8(v3_18, v3_26);
      const __m256i v2_22 = _mm256_unpacklo_epi8(v3_19, v3_27);
      const __m256i v2_23 = _mm256_unpackhi_epi8(v3_19, v3_27);
      const __m256i v2_24 = _mm256_unpacklo_epi8(v3_20, v3_28);
      const __m256i v2_25 = _mm256_unpackhi_epi8(v3_20, v3_28);
      const __m256i v2_26 = _mm256_unpacklo_epi8(v3_21, v3_29);
      const __m256i v2_27 = _mm256_unpackhi_epi8(v3_21, v3_29);
      const __m256i v2_28 = _mm256_unpacklo_epi8(v3_22, v3_30);
      const __m256i v2_29 = _mm256_unpackhi_epi8(v3_22, v3_30);
      const __m256i v2_30 = _mm256_unpacklo_epi8(v3_23, v3_31);
      const __m256i v2_31 = _mm256_unpackhi_epi8(v3_23, v3_31);
      const __m256i v1_0 = _mm256_unpacklo_epi8(v2_0, v2_8);
      const __m256i v1_1 = _mm256_unpackhi_epi8(v2_0, v2_8);
      const __m256i v1_2 = _mm256_unpacklo_epi8(v2_1, v2_9);
      const __m256i v1_3 = _mm256_unpackhi_epi8(v2_1, v2_9);
      const __m256i v1_4 = _mm256_unpacklo_epi8(v2_2, v2_10);
      const __m256i v1_5 = _mm256_unpackhi_epi8(v2_2, v2_10);
      const __m256i v1_6 = _mm256_unpacklo_epi8(v2_3, v2_11);
      const __m256i v1_7 = _mm256_unpackhi_epi8(v2_3, v2_11);
      const __m256i v1_8 = _mm256_unpacklo_epi8(v2_4, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi8(v2_4, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi8(v2_5, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi8(v2_5, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi8(v2_6, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi8(v2_6, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi8(v2_7, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi8(v2_7, v2_15);
      const __m256i v1_16 = _mm256_unpacklo_epi8(v2_16, v2_24);
      const __m256i v1_17 = _mm256_unpackhi_epi8(v2_16, v2_24);
      const __m256i v1_18 = _mm256_unpacklo_epi8(v2_17, v2_25);
      const __m256i v1_19 = _mm256_unpackhi_epi8(v2_17, v2_25);
      const __m256i v1_20 = _mm256_unpacklo_epi8(v2_18, v2_26);
      const __m256i v1_21 = _mm256_unpackhi_epi8(v2_18, v2_26);
      const __m256i v1_22 = _mm256_unpacklo_epi8(v2_19, v2_27);
      const __m256i v1_23 = _mm256_unpackhi_epi8(v2_19, v2_27);
      const __m256i v1_24 = _mm256_unpacklo_epi8(v2_20, v2_28);
      const __m256i v1_25 = _mm256_unpackhi_epi8(v2_20, v2_28);
      const __m256i v1_26 = _mm256_unpacklo_epi8(v2_21, v2_29);
      const __m256i v1_27 = _mm256_unpackhi_epi8(v2_21, v2_29);
      const __m256i v1_28 = _mm256_unpacklo_epi8(v2_22, v2_30);
      const __m256i v1_29 = _mm256_unpackhi_epi8(v2_22, v2_30);
      const __m256i v1_30 = _mm256_unpacklo_epi8(v2_23, v2_31);
      const __m256i v1_31 = _mm256_unpackhi_epi8(v2_23, v2_31);

      const __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_16, 0x20);
      const __m256i v0_16 = _mm256_permute2f128_ps(v1_0, v1_16, 0x31);
      const __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_17, 0x20);
      const __m256i v0_17 = _mm256_permute2f128_ps(v1_1, v1_17, 0x31);
      const __m256i v0_2 = _mm256_permute2f128_ps(v1_2, v1_18, 0x20);
      const __m256i v0_18 = _mm256_permute2f128_ps(v1_2, v1_18, 0x31);
      const __m256i v0_3 = _mm256_permute2f128_ps(v1_3, v1_19, 0x20);
      const __m256i v0_19 = _mm256_permute2f128_ps(v1_3, v1_19, 0x31);
      const __m256i v0_4 = _mm256_permute2f128_ps(v1_4, v1_20, 0x20);
      const __m256i v0_20 = _mm256_permute2f128_ps(v1_4, v1_20, 0x31);
      const __m256i v0_5 = _mm256_permute2f128_ps(v1_5, v1_21, 0x20);
      const __m256i v0_21 = _mm256_permute2f128_ps(v1_5, v1_21, 0x31);
      const __m256i v0_6 = _mm256_permute2f128_ps(v1_6, v1_22, 0x20);
      const __m256i v0_22 = _mm256_permute2f128_ps(v1_6, v1_22, 0x31);
      const __m256i v0_7 = _mm256_permute2f128_ps(v1_7, v1_23, 0x20);
      const __m256i v0_23 = _mm256_permute2f128_ps(v1_7, v1_23, 0x31);
      const __m256i v0_8 = _mm256_permute2f128_ps(v1_8, v1_24, 0x20);
      const __m256i v0_24 = _mm256_permute2f128_ps(v1_8, v1_24, 0x31);
      const __m256i v0_9 = _mm256_permute2f128_ps(v1_9, v1_25, 0x20);
      const __m256i v0_25 = _mm256_permute2f128_ps(v1_9, v1_25, 0x31);
      const __m256i v0_10 = _mm256_permute2f128_ps(v1_10, v1_26, 0x20);
      const __m256i v0_26 = _mm256_permute2f128_ps(v1_10, v1_26, 0x31);
      const __m256i v0_11 = _mm256_permute2f128_ps(v1_11, v1_27, 0x20);
      const __m256i v0_27 = _mm256_permute2f128_ps(v1_11, v1_27, 0x31);
      const __m256i v0_12 = _mm256_permute2f128_ps(v1_12, v1_28, 0x20);
      const __m256i v0_28 = _mm256_permute2f128_ps(v1_12, v1_28, 0x31);
      const __m256i v0_13 = _mm256_permute2f128_ps(v1_13, v1_29, 0x20);
      const __m256i v0_29 = _mm256_permute2f128_ps(v1_13, v1_29, 0x31);
      const __m256i v0_14 = _mm256_permute2f128_ps(v1_14, v1_30, 0x20);
      const __m256i v0_30 = _mm256_permute2f128_ps(v1_14, v1_30, 0x31);
      const __m256i v0_15 = _mm256_permute2f128_ps(v1_15, v1_31, 0x20);
      const __m256i v0_31 = _mm256_permute2f128_ps(v1_15, v1_31, 0x31);

      o = (uint8_t*) ((uintptr_t) o + oN_offset);
      _mm256_storeu_si256((__m256i*) o, v0_31);
      uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 31) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_30);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 31) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_29);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 29) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_28);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 29) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_27);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 27) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_26);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 27) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_25);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 25) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_24);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 25) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_23);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 23) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_22);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 23) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_21);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 21) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_20);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 21) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_19);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 19) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_18);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 19) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_17);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 17) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_16);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 17) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_15);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 15) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_14);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 15) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_13);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 13) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_12);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 13) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_11);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 11) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_10);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 11) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_9);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 9) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_8);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 9) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_7);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 7) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_6);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 7) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_5);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 5) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_4);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 5) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_3);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 3) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_2);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 3) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_1);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_0);
    }
    o = (uint8_t*) ((uintptr_t) o + tile_hbytes);
    if (bh != 0) {
      const __m256i v5_0 = _mm256_maskload_ps((const float*) i0, vmask);
      const uint8_t *i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v5_1 = _mm256_maskload_ps((const float*) i1, vmask);
      const uint8_t *i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v5_2 = _mm256_maskload_ps((const float*) i2, vmask);
      const uint8_t *i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256i v5_3 = _mm256_maskload_ps((const float*) i3, vmask);
      const uint8_t *i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256i v5_4 = _mm256_maskload_ps((const float*) i4, vmask);
      const uint8_t *i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256i v5_5 = _mm256_maskload_ps((const float*) i5, vmask);
      const uint8_t *i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256i v5_6 = _mm256_maskload_ps((const float*) i6, vmask);
      const uint8_t *i7 = (const uint8_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const __m256i v5_7 = _mm256_maskload_ps((const float*) i7, vmask);
      const uint8_t *i8 = (const uint8_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const __m256i v5_8 = _mm256_maskload_ps((const float*) i8, vmask);
      const uint8_t *i9 = (const uint8_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const __m256i v5_9 = _mm256_maskload_ps((const float*) i9, vmask);
      const uint8_t *i10 = (const uint8_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const __m256i v5_10 = _mm256_maskload_ps((const float*) i10, vmask);
      const uint8_t *i11 = (const uint8_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const __m256i v5_11 = _mm256_maskload_ps((const float*) i11, vmask);
      const uint8_t *i12 = (const uint8_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const __m256i v5_12 = _mm256_maskload_ps((const float*) i12, vmask);
      const uint8_t *i13 = (const uint8_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const __m256i v5_13 = _mm256_maskload_ps((const float*) i13, vmask);
      const uint8_t *i14 = (const uint8_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const __m256i v5_14 = _mm256_maskload_ps((const float*) i14, vmask);
      const uint8_t *i15 = (const uint8_t*) ((uintptr_t) i14 + input_stride);
      if XNN_UNPREDICTABLE(bh < 16) {
        i15 = i14;
      }
      const __m256i v5_15 = _mm256_maskload_ps((const float*) i15, vmask);
      const uint8_t *i16 = (const uint8_t*) ((uintptr_t) i15 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 16) {
        i16 = i15;
      }
      const __m256i v5_16 = _mm256_maskload_ps((const float*) i16, vmask);
      const uint8_t *i17 = (const uint8_t*) ((uintptr_t) i16 + input_stride);
      if XNN_UNPREDICTABLE(bh < 18) {
        i17 = i16;
      }
      const __m256i v5_17 = _mm256_maskload_ps((const float*) i17, vmask);
      const uint8_t *i18 = (const uint8_t*) ((uintptr_t) i17 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 18) {
        i18 = i17;
      }
      const __m256i v5_18 = _mm256_maskload_ps((const float*) i18, vmask);
      const uint8_t *i19 = (const uint8_t*) ((uintptr_t) i18 + input_stride);
      if XNN_UNPREDICTABLE(bh < 20) {
        i19 = i18;
      }
      const __m256i v5_19 = _mm256_maskload_ps((const float*) i19, vmask);
      const uint8_t *i20 = (const uint8_t*) ((uintptr_t) i19 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 20) {
        i20 = i19;
      }
      const __m256i v5_20 = _mm256_maskload_ps((const float*) i20, vmask);
      const uint8_t *i21 = (const uint8_t*) ((uintptr_t) i20 + input_stride);
      if XNN_UNPREDICTABLE(bh < 22) {
        i21 = i20;
      }
      const __m256i v5_21 = _mm256_maskload_ps((const float*) i21, vmask);
      const uint8_t *i22 = (const uint8_t*) ((uintptr_t) i21 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 22) {
        i22 = i21;
      }
      const __m256i v5_22 = _mm256_maskload_ps((const float*) i22, vmask);
      const uint8_t *i23 = (const uint8_t*) ((uintptr_t) i22 + input_stride);
      if XNN_UNPREDICTABLE(bh < 24) {
        i23 = i22;
      }
      const __m256i v5_23 = _mm256_maskload_ps((const float*) i23, vmask);
      const uint8_t *i24 = (const uint8_t*) ((uintptr_t) i23 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 24) {
        i24 = i23;
      }
      const __m256i v5_24 = _mm256_maskload_ps((const float*) i24, vmask);
      const uint8_t *i25 = (const uint8_t*) ((uintptr_t) i24 + input_stride);
      if XNN_UNPREDICTABLE(bh < 26) {
        i25 = i24;
      }
      const __m256i v5_25 = _mm256_maskload_ps((const float*) i25, vmask);
      const uint8_t *i26 = (const uint8_t*) ((uintptr_t) i25 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 26) {
        i26 = i25;
      }
      const __m256i v5_26 = _mm256_maskload_ps((const float*) i26, vmask);
      const uint8_t *i27 = (const uint8_t*) ((uintptr_t) i26 + input_stride);
      if XNN_UNPREDICTABLE(bh < 28) {
        i27 = i26;
      }
      const __m256i v5_27 = _mm256_maskload_ps((const float*) i27, vmask);
      const uint8_t *i28 = (const uint8_t*) ((uintptr_t) i27 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 28) {
        i28 = i27;
      }
      const __m256i v5_28 = _mm256_maskload_ps((const float*) i28, vmask);
      const uint8_t *i29 = (const uint8_t*) ((uintptr_t) i28 + input_stride);
      if XNN_UNPREDICTABLE(bh < 30) {
        i29 = i28;
      }
      const __m256i v5_29 = _mm256_maskload_ps((const float*) i29, vmask);
      const uint8_t *i30 = (const uint8_t*) ((uintptr_t) i29 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 30) {
        i30 = i29;
      }
      const __m256i v5_30 = _mm256_maskload_ps((const float*) i30, vmask);
      const __m256i v5_31 = _mm256_undefined_si256();

      const __m256i v4_0 = _mm256_unpacklo_epi8(v5_0, v5_8);
      const __m256i v4_1 = _mm256_unpackhi_epi8(v5_0, v5_8);
      const __m256i v4_2 = _mm256_unpacklo_epi8(v5_1, v5_9);
      const __m256i v4_3 = _mm256_unpackhi_epi8(v5_1, v5_9);
      const __m256i v4_4 = _mm256_unpacklo_epi8(v5_2, v5_10);
      const __m256i v4_5 = _mm256_unpackhi_epi8(v5_2, v5_10);
      const __m256i v4_6 = _mm256_unpacklo_epi8(v5_3, v5_11);
      const __m256i v4_7 = _mm256_unpackhi_epi8(v5_3, v5_11);
      const __m256i v4_8 = _mm256_unpacklo_epi8(v5_4, v5_12);
      const __m256i v4_9 = _mm256_unpackhi_epi8(v5_4, v5_12);
      const __m256i v4_10 = _mm256_unpacklo_epi8(v5_5, v5_13);
      const __m256i v4_11 = _mm256_unpackhi_epi8(v5_5, v5_13);
      const __m256i v4_12 = _mm256_unpacklo_epi8(v5_6, v5_14);
      const __m256i v4_13 = _mm256_unpackhi_epi8(v5_6, v5_14);
      const __m256i v4_14 = _mm256_unpacklo_epi8(v5_7, v5_15);
      const __m256i v4_15 = _mm256_unpackhi_epi8(v5_7, v5_15);
      const __m256i v4_16 = _mm256_unpacklo_epi8(v5_16, v5_24);
      const __m256i v4_17 = _mm256_unpackhi_epi8(v5_16, v5_24);
      const __m256i v4_18 = _mm256_unpacklo_epi8(v5_17, v5_25);
      const __m256i v4_19 = _mm256_unpackhi_epi8(v5_17, v5_25);
      const __m256i v4_20 = _mm256_unpacklo_epi8(v5_18, v5_26);
      const __m256i v4_21 = _mm256_unpackhi_epi8(v5_18, v5_26);
      const __m256i v4_22 = _mm256_unpacklo_epi8(v5_19, v5_27);
      const __m256i v4_23 = _mm256_unpackhi_epi8(v5_19, v5_27);
      const __m256i v4_24 = _mm256_unpacklo_epi8(v5_20, v5_28);
      const __m256i v4_25 = _mm256_unpackhi_epi8(v5_20, v5_28);
      const __m256i v4_26 = _mm256_unpacklo_epi8(v5_21, v5_29);
      const __m256i v4_27 = _mm256_unpackhi_epi8(v5_21, v5_29);
      const __m256i v4_28 = _mm256_unpacklo_epi8(v5_22, v5_30);
      const __m256i v4_29 = _mm256_unpackhi_epi8(v5_22, v5_30);
      const __m256i v4_30 = _mm256_unpacklo_epi8(v5_23, v5_31);
      const __m256i v4_31 = _mm256_unpackhi_epi8(v5_23, v5_31);
      const __m256i v3_0 = _mm256_unpacklo_epi8(v4_0, v4_8);
      const __m256i v3_1 = _mm256_unpackhi_epi8(v4_0, v4_8);
      const __m256i v3_2 = _mm256_unpacklo_epi8(v4_1, v4_9);
      const __m256i v3_3 = _mm256_unpackhi_epi8(v4_1, v4_9);
      const __m256i v3_4 = _mm256_unpacklo_epi8(v4_2, v4_10);
      const __m256i v3_5 = _mm256_unpackhi_epi8(v4_2, v4_10);
      const __m256i v3_6 = _mm256_unpacklo_epi8(v4_3, v4_11);
      const __m256i v3_7 = _mm256_unpackhi_epi8(v4_3, v4_11);
      const __m256i v3_8 = _mm256_unpacklo_epi8(v4_4, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi8(v4_4, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi8(v4_5, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi8(v4_5, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi8(v4_6, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi8(v4_6, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi8(v4_7, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi8(v4_7, v4_15);
      const __m256i v3_16 = _mm256_unpacklo_epi8(v4_16, v4_24);
      const __m256i v3_17 = _mm256_unpackhi_epi8(v4_16, v4_24);
      const __m256i v3_18 = _mm256_unpacklo_epi8(v4_17, v4_25);
      const __m256i v3_19 = _mm256_unpackhi_epi8(v4_17, v4_25);
      const __m256i v3_20 = _mm256_unpacklo_epi8(v4_18, v4_26);
      const __m256i v3_21 = _mm256_unpackhi_epi8(v4_18, v4_26);
      const __m256i v3_22 = _mm256_unpacklo_epi8(v4_19, v4_27);
      const __m256i v3_23 = _mm256_unpackhi_epi8(v4_19, v4_27);
      const __m256i v3_24 = _mm256_unpacklo_epi8(v4_20, v4_28);
      const __m256i v3_25 = _mm256_unpackhi_epi8(v4_20, v4_28);
      const __m256i v3_26 = _mm256_unpacklo_epi8(v4_21, v4_29);
      const __m256i v3_27 = _mm256_unpackhi_epi8(v4_21, v4_29);
      const __m256i v3_28 = _mm256_unpacklo_epi8(v4_22, v4_30);
      const __m256i v3_29 = _mm256_unpackhi_epi8(v4_22, v4_30);
      const __m256i v3_30 = _mm256_unpacklo_epi8(v4_23, v4_31);
      const __m256i v3_31 = _mm256_unpackhi_epi8(v4_23, v4_31);
      const __m256i v2_0 = _mm256_unpacklo_epi8(v3_0, v3_8);
      const __m256i v2_1 = _mm256_unpackhi_epi8(v3_0, v3_8);
      const __m256i v2_2 = _mm256_unpacklo_epi8(v3_1, v3_9);
      const __m256i v2_3 = _mm256_unpackhi_epi8(v3_1, v3_9);
      const __m256i v2_4 = _mm256_unpacklo_epi8(v3_2, v3_10);
      const __m256i v2_5 = _mm256_unpackhi_epi8(v3_2, v3_10);
      const __m256i v2_6 = _mm256_unpacklo_epi8(v3_3, v3_11);
      const __m256i v2_7 = _mm256_unpackhi_epi8(v3_3, v3_11);
      const __m256i v2_8 = _mm256_unpacklo_epi8(v3_4, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi8(v3_4, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi8(v3_5, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi8(v3_5, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi8(v3_6, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi8(v3_6, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi8(v3_7, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi8(v3_7, v3_15);
      const __m256i v2_16 = _mm256_unpacklo_epi8(v3_16, v3_24);
      const __m256i v2_17 = _mm256_unpackhi_epi8(v3_16, v3_24);
      const __m256i v2_18 = _mm256_unpacklo_epi8(v3_17, v3_25);
      const __m256i v2_19 = _mm256_unpackhi_epi8(v3_17, v3_25);
      const __m256i v2_20 = _mm256_unpacklo_epi8(v3_18, v3_26);
      const __m256i v2_21 = _mm256_unpackhi_epi8(v3_18, v3_26);
      const __m256i v2_22 = _mm256_unpacklo_epi8(v3_19, v3_27);
      const __m256i v2_23 = _mm256_unpackhi_epi8(v3_19, v3_27);
      const __m256i v2_24 = _mm256_unpacklo_epi8(v3_20, v3_28);
      const __m256i v2_25 = _mm256_unpackhi_epi8(v3_20, v3_28);
      const __m256i v2_26 = _mm256_unpacklo_epi8(v3_21, v3_29);
      const __m256i v2_27 = _mm256_unpackhi_epi8(v3_21, v3_29);
      const __m256i v2_28 = _mm256_unpacklo_epi8(v3_22, v3_30);
      const __m256i v2_29 = _mm256_unpackhi_epi8(v3_22, v3_30);
      const __m256i v2_30 = _mm256_unpacklo_epi8(v3_23, v3_31);
      const __m256i v2_31 = _mm256_unpackhi_epi8(v3_23, v3_31);
      const __m256i v1_0 = _mm256_unpacklo_epi8(v2_0, v2_8);
      const __m256i v1_1 = _mm256_unpackhi_epi8(v2_0, v2_8);
      const __m256i v1_2 = _mm256_unpacklo_epi8(v2_1, v2_9);
      const __m256i v1_3 = _mm256_unpackhi_epi8(v2_1, v2_9);
      const __m256i v1_4 = _mm256_unpacklo_epi8(v2_2, v2_10);
      const __m256i v1_5 = _mm256_unpackhi_epi8(v2_2, v2_10);
      const __m256i v1_6 = _mm256_unpacklo_epi8(v2_3, v2_11);
      const __m256i v1_7 = _mm256_unpackhi_epi8(v2_3, v2_11);
      const __m256i v1_8 = _mm256_unpacklo_epi8(v2_4, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi8(v2_4, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi8(v2_5, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi8(v2_5, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi8(v2_6, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi8(v2_6, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi8(v2_7, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi8(v2_7, v2_15);
      const __m256i v1_16 = _mm256_unpacklo_epi8(v2_16, v2_24);
      const __m256i v1_17 = _mm256_unpackhi_epi8(v2_16, v2_24);
      const __m256i v1_18 = _mm256_unpacklo_epi8(v2_17, v2_25);
      const __m256i v1_19 = _mm256_unpackhi_epi8(v2_17, v2_25);
      const __m256i v1_20 = _mm256_unpacklo_epi8(v2_18, v2_26);
      const __m256i v1_21 = _mm256_unpackhi_epi8(v2_18, v2_26);
      const __m256i v1_22 = _mm256_unpacklo_epi8(v2_19, v2_27);
      const __m256i v1_23 = _mm256_unpackhi_epi8(v2_19, v2_27);
      const __m256i v1_24 = _mm256_unpacklo_epi8(v2_20, v2_28);
      const __m256i v1_25 = _mm256_unpackhi_epi8(v2_20, v2_28);
      const __m256i v1_26 = _mm256_unpacklo_epi8(v2_21, v2_29);
      const __m256i v1_27 = _mm256_unpackhi_epi8(v2_21, v2_29);
      const __m256i v1_28 = _mm256_unpacklo_epi8(v2_22, v2_30);
      const __m256i v1_29 = _mm256_unpackhi_epi8(v2_22, v2_30);
      const __m256i v1_30 = _mm256_unpacklo_epi8(v2_23, v2_31);
      const __m256i v1_31 = _mm256_unpackhi_epi8(v2_23, v2_31);

      __m256i v0_0 = _mm256_permute2f128_ps(v1_0, v1_16, 0x20);
      __m256i v0_16 = _mm256_permute2f128_ps(v1_0, v1_16, 0x31);
      __m256i v0_1 = _mm256_permute2f128_ps(v1_1, v1_17, 0x20);
      __m256i v0_17 = _mm256_permute2f128_ps(v1_1, v1_17, 0x31);
      __m256i v0_2 = _mm256_permute2f128_ps(v1_2, v1_18, 0x20);
      __m256i v0_18 = _mm256_permute2f128_ps(v1_2, v1_18, 0x31);
      __m256i v0_3 = _mm256_permute2f128_ps(v1_3, v1_19, 0x20);
      __m256i v0_19 = _mm256_permute2f128_ps(v1_3, v1_19, 0x31);
      __m256i v0_4 = _mm256_permute2f128_ps(v1_4, v1_20, 0x20);
      __m256i v0_20 = _mm256_permute2f128_ps(v1_4, v1_20, 0x31);
      __m256i v0_5 = _mm256_permute2f128_ps(v1_5, v1_21, 0x20);
      __m256i v0_21 = _mm256_permute2f128_ps(v1_5, v1_21, 0x31);
      __m256i v0_6 = _mm256_permute2f128_ps(v1_6, v1_22, 0x20);
      __m256i v0_22 = _mm256_permute2f128_ps(v1_6, v1_22, 0x31);
      __m256i v0_7 = _mm256_permute2f128_ps(v1_7, v1_23, 0x20);
      __m256i v0_23 = _mm256_permute2f128_ps(v1_7, v1_23, 0x31);
      __m256i v0_8 = _mm256_permute2f128_ps(v1_8, v1_24, 0x20);
      __m256i v0_24 = _mm256_permute2f128_ps(v1_8, v1_24, 0x31);
      __m256i v0_9 = _mm256_permute2f128_ps(v1_9, v1_25, 0x20);
      __m256i v0_25 = _mm256_permute2f128_ps(v1_9, v1_25, 0x31);
      __m256i v0_10 = _mm256_permute2f128_ps(v1_10, v1_26, 0x20);
      __m256i v0_26 = _mm256_permute2f128_ps(v1_10, v1_26, 0x31);
      __m256i v0_11 = _mm256_permute2f128_ps(v1_11, v1_27, 0x20);
      __m256i v0_27 = _mm256_permute2f128_ps(v1_11, v1_27, 0x31);
      __m256i v0_12 = _mm256_permute2f128_ps(v1_12, v1_28, 0x20);
      __m256i v0_28 = _mm256_permute2f128_ps(v1_12, v1_28, 0x31);
      __m256i v0_13 = _mm256_permute2f128_ps(v1_13, v1_29, 0x20);
      __m256i v0_29 = _mm256_permute2f128_ps(v1_13, v1_29, 0x31);
      __m256i v0_14 = _mm256_permute2f128_ps(v1_14, v1_30, 0x20);
      __m256i v0_30 = _mm256_permute2f128_ps(v1_14, v1_30, 0x31);
      __m256i v0_15 = _mm256_permute2f128_ps(v1_15, v1_31, 0x20);
      __m256i v0_31 = _mm256_permute2f128_ps(v1_15, v1_31, 0x31);


      if (bh & 16) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_31));
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_30));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_29));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_28));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_27));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_26));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_25));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_24));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_23));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_22));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_21));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_20));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_19));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_18));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_17));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_16));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_15));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_14));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_13));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_12));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_11));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_10));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_9));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_8));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_7));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_6));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_5));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_4));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_3));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_2));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_1));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, _mm256_castsi256_si128(v0_0));
        o += 16;
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
        v0_16 = _mm256_permute2f128_ps(v0_16, v0_16, 0x1);
        v0_17 = _mm256_permute2f128_ps(v0_17, v0_17, 0x1);
        v0_18 = _mm256_permute2f128_ps(v0_18, v0_18, 0x1);
        v0_19 = _mm256_permute2f128_ps(v0_19, v0_19, 0x1);
        v0_20 = _mm256_permute2f128_ps(v0_20, v0_20, 0x1);
        v0_21 = _mm256_permute2f128_ps(v0_21, v0_21, 0x1);
        v0_22 = _mm256_permute2f128_ps(v0_22, v0_22, 0x1);
        v0_23 = _mm256_permute2f128_ps(v0_23, v0_23, 0x1);
        v0_24 = _mm256_permute2f128_ps(v0_24, v0_24, 0x1);
        v0_25 = _mm256_permute2f128_ps(v0_25, v0_25, 0x1);
        v0_26 = _mm256_permute2f128_ps(v0_26, v0_26, 0x1);
        v0_27 = _mm256_permute2f128_ps(v0_27, v0_27, 0x1);
        v0_28 = _mm256_permute2f128_ps(v0_28, v0_28, 0x1);
        v0_29 = _mm256_permute2f128_ps(v0_29, v0_29, 0x1);
        v0_30 = _mm256_permute2f128_ps(v0_30, v0_30, 0x1);
        v0_31 = _mm256_permute2f128_ps(v0_31, v0_31, 0x1);
      }

      if (bh & 8) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_31));
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_30));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_29));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_28));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_27));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_26));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_25));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_24));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_23));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_22));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_21));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_20));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_19));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_18));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_17));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_16));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_15));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_14));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_13));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_12));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_11));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_10));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_9));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_8));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_7));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_6));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_5));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_4));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_3));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_2));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_1));
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, _mm256_castsi256_si128(v0_0));
        o += 8;
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
        v0_16 = _mm256_unpackhi_pd(v0_16, v0_16);
        v0_17 = _mm256_unpackhi_pd(v0_17, v0_17);
        v0_18 = _mm256_unpackhi_pd(v0_18, v0_18);
        v0_19 = _mm256_unpackhi_pd(v0_19, v0_19);
        v0_20 = _mm256_unpackhi_pd(v0_20, v0_20);
        v0_21 = _mm256_unpackhi_pd(v0_21, v0_21);
        v0_22 = _mm256_unpackhi_pd(v0_22, v0_22);
        v0_23 = _mm256_unpackhi_pd(v0_23, v0_23);
        v0_24 = _mm256_unpackhi_pd(v0_24, v0_24);
        v0_25 = _mm256_unpackhi_pd(v0_25, v0_25);
        v0_26 = _mm256_unpackhi_pd(v0_26, v0_26);
        v0_27 = _mm256_unpackhi_pd(v0_27, v0_27);
        v0_28 = _mm256_unpackhi_pd(v0_28, v0_28);
        v0_29 = _mm256_unpackhi_pd(v0_29, v0_29);
        v0_30 = _mm256_unpackhi_pd(v0_30, v0_30);
        v0_31 = _mm256_unpackhi_pd(v0_31, v0_31);
      }
      if (bh & 4) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        *((int*) o) = _mm256_cvtsi256_si32(v0_31);
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_30);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_29);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_28);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_27);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_26);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_25);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_24);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_23);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_22);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_21);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_20);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_19);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_18);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_17);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_15);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_14);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_13);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_12);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_11);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_10);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_9);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_8);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_7);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_6);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_5);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_4);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_3);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_2);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        *((int*) o) = _mm256_cvtsi256_si32(v0_0);
        o += 4;
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
        v0_16 = _mm256_srli_epi64(v0_16, 32);
        v0_17 = _mm256_srli_epi64(v0_17, 32);
        v0_18 = _mm256_srli_epi64(v0_18, 32);
        v0_19 = _mm256_srli_epi64(v0_19, 32);
        v0_20 = _mm256_srli_epi64(v0_20, 32);
        v0_21 = _mm256_srli_epi64(v0_21, 32);
        v0_22 = _mm256_srli_epi64(v0_22, 32);
        v0_23 = _mm256_srli_epi64(v0_23, 32);
        v0_24 = _mm256_srli_epi64(v0_24, 32);
        v0_25 = _mm256_srli_epi64(v0_25, 32);
        v0_26 = _mm256_srli_epi64(v0_26, 32);
        v0_27 = _mm256_srli_epi64(v0_27, 32);
        v0_28 = _mm256_srli_epi64(v0_28, 32);
        v0_29 = _mm256_srli_epi64(v0_29, 32);
        v0_30 = _mm256_srli_epi64(v0_30, 32);
        v0_31 = _mm256_srli_epi64(v0_31, 32);
      }
      if (bh & 2) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_31);
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_30);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_29);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_28);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_27);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_26);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_25);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_24);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_23);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_22);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_21);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_20);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_19);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_18);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_17);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_15);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_14);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_13);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_12);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_11);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_10);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_9);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_8);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_7);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_6);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_5);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_4);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_3);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_2);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        *((uint16_t*) o) = (uint16_t) _mm256_cvtsi256_si32(v0_0);
        o += 2;
      }
      v0_0 = _mm256_srli_epi32(v0_0, 16);
      v0_1 = _mm256_srli_epi32(v0_1, 16);
      v0_2 = _mm256_srli_epi32(v0_2, 16);
      v0_3 = _mm256_srli_epi32(v0_3, 16);
      v0_4 = _mm256_srli_epi32(v0_4, 16);
      v0_5 = _mm256_srli_epi32(v0_5, 16);
      v0_6 = _mm256_srli_epi32(v0_6, 16);
      v0_7 = _mm256_srli_epi32(v0_7, 16);
      v0_8 = _mm256_srli_epi32(v0_8, 16);
      v0_9 = _mm256_srli_epi32(v0_9, 16);
      v0_10 = _mm256_srli_epi32(v0_10, 16);
      v0_11 = _mm256_srli_epi32(v0_11, 16);
      v0_12 = _mm256_srli_epi32(v0_12, 16);
      v0_13 = _mm256_srli_epi32(v0_13, 16);
      v0_14 = _mm256_srli_epi32(v0_14, 16);
      v0_15 = _mm256_srli_epi32(v0_15, 16);
      v0_16 = _mm256_srli_epi32(v0_16, 16);
      v0_17 = _mm256_srli_epi32(v0_17, 16);
      v0_18 = _mm256_srli_epi32(v0_18, 16);
      v0_19 = _mm256_srli_epi32(v0_19, 16);
      v0_20 = _mm256_srli_epi32(v0_20, 16);
      v0_21 = _mm256_srli_epi32(v0_21, 16);
      v0_22 = _mm256_srli_epi32(v0_22, 16);
      v0_23 = _mm256_srli_epi32(v0_23, 16);
      v0_24 = _mm256_srli_epi32(v0_24, 16);
      v0_25 = _mm256_srli_epi32(v0_25, 16);
      v0_26 = _mm256_srli_epi32(v0_26, 16);
      v0_27 = _mm256_srli_epi32(v0_27, 16);
      v0_28 = _mm256_srli_epi32(v0_28, 16);
      v0_29 = _mm256_srli_epi32(v0_29, 16);
      v0_30 = _mm256_srli_epi32(v0_30, 16);
      v0_31 = _mm256_srli_epi32(v0_31, 16);
      if (bh & 1) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_31);
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_30);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_29);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_28);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_27);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_26);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_25);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_24);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_23);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_22);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_21);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_20);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_19);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_18);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_17);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_15);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_14);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_13);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_12);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_11);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_10);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_9);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_8);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_7);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_6);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_5);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_4);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_3);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_2);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        *o = (uint8_t) _mm256_cvtsi256_si32(v0_0);
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o = (uint8_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
