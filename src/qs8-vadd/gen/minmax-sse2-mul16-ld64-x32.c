// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/sse-mul16-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/vadd.h>


void xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_x32(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const __m128i vzero_point_product = _mm_load_si128((const __m128i*) params->sse2.zero_point_product);
  const __m128i va_multiplier_lo = _mm_load_si128((const __m128i*) params->sse2.x_multiplier_lo);
  const __m128i va_multiplier_hi = _mm_load_si128((const __m128i*) params->sse2.x_multiplier_hi);
  const __m128i vb_multiplier_lo = _mm_load_si128((const __m128i*) params->sse2.y_multiplier_lo);
  const __m128i vb_multiplier_hi = _mm_load_si128((const __m128i*) params->sse2.y_multiplier_hi);
  const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->sse2.remainder_mask);
  const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->sse2.remainder_threshold);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->sse2.shift);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);

  for (; n >= 32 * sizeof(int8_t); n -= 32 * sizeof(int8_t)) {
    __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
    __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);
    __m128i va89ABCDEF = _mm_loadl_epi64((const __m128i*) (input_a + 8));
    __m128i vb89ABCDEF = _mm_loadl_epi64((const __m128i*) (input_b + 8));
    __m128i vaGHIJKLMN = _mm_loadl_epi64((const __m128i*) (input_a + 16));
    __m128i vbGHIJKLMN = _mm_loadl_epi64((const __m128i*) (input_b + 16));
    __m128i vaOPQRSTUV = _mm_loadl_epi64((const __m128i*) (input_a + 24));
    __m128i vbOPQRSTUV = _mm_loadl_epi64((const __m128i*) (input_b + 24));
    input_a += 32;
    input_b += 32;

    va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);
    vb01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vb01234567, vb01234567), 8);
    va89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(va89ABCDEF, va89ABCDEF), 8);
    vb89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vb89ABCDEF, vb89ABCDEF), 8);
    vaGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vaGHIJKLMN, vaGHIJKLMN), 8);
    vbGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vbGHIJKLMN, vbGHIJKLMN), 8);
    vaOPQRSTUV = _mm_srai_epi16(_mm_unpacklo_epi8(vaOPQRSTUV, vaOPQRSTUV), 8);
    vbOPQRSTUV = _mm_srai_epi16(_mm_unpacklo_epi8(vbOPQRSTUV, vbOPQRSTUV), 8);

    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
    const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);
    __m128i vaprod89ABCDEFhi = _mm_mulhi_epu16(va89ABCDEF, va_multiplier_lo);
    __m128i vbprod89ABCDEFhi = _mm_mulhi_epu16(vb89ABCDEF, vb_multiplier_lo);
    const __m128i vaprod89ABCDEFlo = _mm_mullo_epi16(va89ABCDEF, va_multiplier_lo);
    const __m128i vbprod89ABCDEFlo = _mm_mullo_epi16(vb89ABCDEF, vb_multiplier_lo);
    __m128i vaprodGHIJKLMNhi = _mm_mulhi_epu16(vaGHIJKLMN, va_multiplier_lo);
    __m128i vbprodGHIJKLMNhi = _mm_mulhi_epu16(vbGHIJKLMN, vb_multiplier_lo);
    const __m128i vaprodGHIJKLMNlo = _mm_mullo_epi16(vaGHIJKLMN, va_multiplier_lo);
    const __m128i vbprodGHIJKLMNlo = _mm_mullo_epi16(vbGHIJKLMN, vb_multiplier_lo);
    __m128i vaprodOPQRSTUVhi = _mm_mulhi_epu16(vaOPQRSTUV, va_multiplier_lo);
    __m128i vbprodOPQRSTUVhi = _mm_mulhi_epu16(vbOPQRSTUV, vb_multiplier_lo);
    const __m128i vaprodOPQRSTUVlo = _mm_mullo_epi16(vaOPQRSTUV, va_multiplier_lo);
    const __m128i vbprodOPQRSTUVlo = _mm_mullo_epi16(vbOPQRSTUV, vb_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
    vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));
    vaprod89ABCDEFhi = _mm_add_epi16(vaprod89ABCDEFhi, _mm_mullo_epi16(va89ABCDEF, va_multiplier_hi));
    vbprod89ABCDEFhi = _mm_add_epi16(vbprod89ABCDEFhi, _mm_mullo_epi16(vb89ABCDEF, vb_multiplier_hi));
    vaprodGHIJKLMNhi = _mm_add_epi16(vaprodGHIJKLMNhi, _mm_mullo_epi16(vaGHIJKLMN, va_multiplier_hi));
    vbprodGHIJKLMNhi = _mm_add_epi16(vbprodGHIJKLMNhi, _mm_mullo_epi16(vbGHIJKLMN, vb_multiplier_hi));
    vaprodOPQRSTUVhi = _mm_add_epi16(vaprodOPQRSTUVhi, _mm_mullo_epi16(vaOPQRSTUV, va_multiplier_hi));
    vbprodOPQRSTUVhi = _mm_add_epi16(vbprodOPQRSTUVhi, _mm_mullo_epi16(vbOPQRSTUV, vb_multiplier_hi));

    vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));
    vbprod01234567hi = _mm_sub_epi16(vbprod01234567hi, _mm_and_si128(_mm_srai_epi16(vb01234567, 15), vb_multiplier_lo));
    vaprod89ABCDEFhi = _mm_sub_epi16(vaprod89ABCDEFhi, _mm_and_si128(_mm_srai_epi16(va89ABCDEF, 15), va_multiplier_lo));
    vbprod89ABCDEFhi = _mm_sub_epi16(vbprod89ABCDEFhi, _mm_and_si128(_mm_srai_epi16(vb89ABCDEF, 15), vb_multiplier_lo));
    vaprodGHIJKLMNhi = _mm_sub_epi16(vaprodGHIJKLMNhi, _mm_and_si128(_mm_srai_epi16(vaGHIJKLMN, 15), va_multiplier_lo));
    vbprodGHIJKLMNhi = _mm_sub_epi16(vbprodGHIJKLMNhi, _mm_and_si128(_mm_srai_epi16(vbGHIJKLMN, 15), vb_multiplier_lo));
    vaprodOPQRSTUVhi = _mm_sub_epi16(vaprodOPQRSTUVhi, _mm_and_si128(_mm_srai_epi16(vaOPQRSTUV, 15), va_multiplier_lo));
    vbprodOPQRSTUVhi = _mm_sub_epi16(vbprodOPQRSTUVhi, _mm_and_si128(_mm_srai_epi16(vbOPQRSTUV, 15), vb_multiplier_lo));

    __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc89AB = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vaprod89ABCDEFlo, vaprod89ABCDEFhi));
    __m128i vaccCDEF = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vaprod89ABCDEFlo, vaprod89ABCDEFhi));
    __m128i vaccGHIJ = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vaprodGHIJKLMNlo, vaprodGHIJKLMNhi));
    __m128i vaccKLMN = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vaprodGHIJKLMNlo, vaprodGHIJKLMNhi));
    __m128i vaccOPQR = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vaprodOPQRSTUVlo, vaprodOPQRSTUVhi));
    __m128i vaccSTUV = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vaprodOPQRSTUVlo, vaprodOPQRSTUVhi));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));
    vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vbprod89ABCDEFlo, vbprod89ABCDEFhi));
    vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vbprod89ABCDEFlo, vbprod89ABCDEFhi));
    vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vbprodGHIJKLMNlo, vbprodGHIJKLMNhi));
    vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vbprodGHIJKLMNlo, vbprodGHIJKLMNhi));
    vaccOPQR = _mm_add_epi32(vaccOPQR, _mm_unpacklo_epi16(vbprodOPQRSTUVlo, vbprodOPQRSTUVhi));
    vaccSTUV = _mm_add_epi32(vaccSTUV, _mm_unpackhi_epi16(vbprodOPQRSTUVlo, vbprodOPQRSTUVhi));

    const __m128i vrem0123 = _mm_add_epi32(_mm_and_si128(vacc0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123));
    const __m128i vrem4567 = _mm_add_epi32(_mm_and_si128(vacc4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567));
    const __m128i vrem89AB = _mm_add_epi32(_mm_and_si128(vacc89AB, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc89AB));
    const __m128i vremCDEF = _mm_add_epi32(_mm_and_si128(vaccCDEF, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccCDEF));
    const __m128i vremGHIJ = _mm_add_epi32(_mm_and_si128(vaccGHIJ, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccGHIJ));
    const __m128i vremKLMN = _mm_add_epi32(_mm_and_si128(vaccKLMN, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccKLMN));
    const __m128i vremOPQR = _mm_add_epi32(_mm_and_si128(vaccOPQR, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccOPQR));
    const __m128i vremSTUV = _mm_add_epi32(_mm_and_si128(vaccSTUV, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vaccSTUV));

    vacc0123 = _mm_sub_epi32(_mm_sra_epi32(vacc0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
    vacc4567 = _mm_sub_epi32(_mm_sra_epi32(vacc4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));
    vacc89AB = _mm_sub_epi32(_mm_sra_epi32(vacc89AB, vshift), _mm_cmpgt_epi32(vrem89AB, vremainder_threshold));
    vaccCDEF = _mm_sub_epi32(_mm_sra_epi32(vaccCDEF, vshift), _mm_cmpgt_epi32(vremCDEF, vremainder_threshold));
    vaccGHIJ = _mm_sub_epi32(_mm_sra_epi32(vaccGHIJ, vshift), _mm_cmpgt_epi32(vremGHIJ, vremainder_threshold));
    vaccKLMN = _mm_sub_epi32(_mm_sra_epi32(vaccKLMN, vshift), _mm_cmpgt_epi32(vremKLMN, vremainder_threshold));
    vaccOPQR = _mm_sub_epi32(_mm_sra_epi32(vaccOPQR, vshift), _mm_cmpgt_epi32(vremOPQR, vremainder_threshold));
    vaccSTUV = _mm_sub_epi32(_mm_sra_epi32(vaccSTUV, vshift), _mm_cmpgt_epi32(vremSTUV, vremainder_threshold));

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);
    __m128i voutGHIJKLMN = _mm_adds_epi16(_mm_packs_epi32(vaccGHIJ, vaccKLMN), voutput_zero_point);
    __m128i voutOPQRSTUV = _mm_adds_epi16(_mm_packs_epi32(vaccOPQR, vaccSTUV), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
    vout89ABCDEF = _mm_max_epi16(vout89ABCDEF, voutput_min);
    voutGHIJKLMN = _mm_max_epi16(voutGHIJKLMN, voutput_min);
    voutOPQRSTUV = _mm_max_epi16(voutOPQRSTUV, voutput_min);

    vout01234567 = _mm_min_epi16(vout01234567, voutput_max);
    vout89ABCDEF = _mm_min_epi16(vout89ABCDEF, voutput_max);
    voutGHIJKLMN = _mm_min_epi16(voutGHIJKLMN, voutput_max);
    voutOPQRSTUV = _mm_min_epi16(voutOPQRSTUV, voutput_max);

    const __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
    const __m128i voutGHIJKLMNOPQRSTUV = _mm_packs_epi16(voutGHIJKLMN, voutOPQRSTUV);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
    output += 32;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      __m128i va01234567 = _mm_loadl_epi64((const __m128i*) input_a);
      __m128i vb01234567 = _mm_loadl_epi64((const __m128i*) input_b);
      input_a += 8;
      input_b += 8;

      va01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(va01234567, va01234567), 8);
      vb01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vb01234567, vb01234567), 8);

      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
      const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
      vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));

      vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));
      vbprod01234567hi = _mm_sub_epi16(vbprod01234567hi, _mm_and_si128(_mm_srai_epi16(vb01234567, 15), vb_multiplier_lo));

      __m128i vacc0123 = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

      const __m128i vrem0123 = _mm_add_epi32(_mm_and_si128(vacc0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123));
      const __m128i vrem4567 = _mm_add_epi32(_mm_and_si128(vacc4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567));

      vacc0123 = _mm_sub_epi32(_mm_sra_epi32(vacc0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
      vacc4567 = _mm_sub_epi32(_mm_sra_epi32(vacc4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if XNN_LIKELY(n >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        n -= 8 * sizeof(int8_t);
      } else {
        if (n & (4 * sizeof(int8_t))) {
          *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (n & (2 * sizeof(int8_t))) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (n & (1 * sizeof(int8_t))) {
          *output = (int32_t) _mm_cvtsi128_si32(vout0123456701234567);
        }
        n = 0;
      }
    } while (n != 0);
  }
}
