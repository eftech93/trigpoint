#!/bin/sh
# Architecture flags shared by csrc/build.sh and pgext/Makefile (one place for the detection):
# arm64 (Apple Silicon): NEON + dotprod; x86-64: -march=native (AVX2 + FMA on Haswell or later).
case "$(uname -m)" in
  arm64|aarch64) echo "-march=armv8.2-a+dotprod" ;;
  *) echo "-march=native" ;;
esac
