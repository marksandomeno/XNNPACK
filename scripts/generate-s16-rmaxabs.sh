#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/s16-rmaxabs/scalar.c.in -D CHANNEL_TILE=1 -o src/s16-rmaxabs/gen/scalar-x1.c &
tools/xngen src/s16-rmaxabs/scalar.c.in -D CHANNEL_TILE=2 -o src/s16-rmaxabs/gen/scalar-x2.c &
tools/xngen src/s16-rmaxabs/scalar.c.in -D CHANNEL_TILE=3 -o src/s16-rmaxabs/gen/scalar-x3.c &
tools/xngen src/s16-rmaxabs/scalar.c.in -D CHANNEL_TILE=4 -o src/s16-rmaxabs/gen/scalar-x4.c &

################################### NEON ###################################
tools/xngen src/s16-rmaxabs/neon.c.in -D CHANNEL_TILE=8  -o src/s16-rmaxabs/gen/neon-x8.c &
tools/xngen src/s16-rmaxabs/neon.c.in -D CHANNEL_TILE=16 -o src/s16-rmaxabs/gen/neon-x16.c &
tools/xngen src/s16-rmaxabs/neon.c.in -D CHANNEL_TILE=24 -o src/s16-rmaxabs/gen/neon-x24.c &
tools/xngen src/s16-rmaxabs/neon.c.in -D CHANNEL_TILE=32 -o src/s16-rmaxabs/gen/neon-x32.c &

################################## Unit tests #################################
tools/generate-rmaxabs-test.py --spec test/s16-rmaxabs.yaml --output test/s16-rmaxabs.cc &

wait
