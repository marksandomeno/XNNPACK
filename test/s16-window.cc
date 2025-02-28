// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-window.yaml
//   Generator: tools/generate-window-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/window.h>
#include "window-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(8)
      .Test(xnn_s16_window_ukernel__neon_x8);
  }

  TEST(S16_WINDOW__NEON_X8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW__NEON_X8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW__NEON_X8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW__NEON_X8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_s16_window_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X8, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(8)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(16)
      .Test(xnn_s16_window_ukernel__neon_x16);
  }

  TEST(S16_WINDOW__NEON_X16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW__NEON_X16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW__NEON_X16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW__NEON_X16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_s16_window_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X16, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(16)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X24, channels_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(24)
      .Test(xnn_s16_window_ukernel__neon_x24);
  }

  TEST(S16_WINDOW__NEON_X24, channels_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 240; channels += 24) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW__NEON_X24, channels_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW__NEON_X24, channels_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW__NEON_X24, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_s16_window_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X24, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(24)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X32, channels_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(32)
      .Test(xnn_s16_window_ukernel__neon_x32);
  }

  TEST(S16_WINDOW__NEON_X32, channels_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 320; channels += 32) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW__NEON_X32, channels_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW__NEON_X32, channels_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW__NEON_X32, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_s16_window_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X32, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(32)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(S16_WINDOW__SCALAR_X1, channels_eq_1) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(1)
    .Test(xnn_s16_window_ukernel__scalar_x1);
}

TEST(S16_WINDOW__SCALAR_X1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x1);
  }
}

TEST(S16_WINDOW__SCALAR_X1, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__scalar_x1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X1, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X1, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(1)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x1);
  }
}


TEST(S16_WINDOW__SCALAR_X2, channels_eq_2) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(2)
    .Test(xnn_s16_window_ukernel__scalar_x2);
}

TEST(S16_WINDOW__SCALAR_X2, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__scalar_x2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X2, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X2, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(2)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}


TEST(S16_WINDOW__SCALAR_X3, channels_eq_3) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(3)
    .Test(xnn_s16_window_ukernel__scalar_x3);
}

TEST(S16_WINDOW__SCALAR_X3, channels_div_3) {
  for (size_t channels = 6; channels < 30; channels += 3) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}

TEST(S16_WINDOW__SCALAR_X3, channels_lt_3) {
  for (size_t channels = 1; channels < 3; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}

TEST(S16_WINDOW__SCALAR_X3, channels_gt_3) {
  for (size_t channels = 4; channels < 6; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}

TEST(S16_WINDOW__SCALAR_X3, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 15; channels += 2) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__scalar_x3);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X3, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 15; channels += 2) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x3);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X3, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(3)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}


TEST(S16_WINDOW__SCALAR_X4, channels_eq_4) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(4)
    .Test(xnn_s16_window_ukernel__scalar_x4);
}

TEST(S16_WINDOW__SCALAR_X4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(channels)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_s16_window_ukernel__scalar_x4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X4, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X4, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(4)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}
