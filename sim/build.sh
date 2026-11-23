#!/usr/bin/env bash
# Build the two C simulation engines as shared libraries (loaded via ctypes).
# Linux / macOS / WSL. Needs a C compiler (gcc or clang).
set -e
cd "$(dirname "$0")"
CC=${CC:-cc}
FLAGS="-O3 -shared -fPIC"
# -march=native is a free ~10-20% on x86; skip it where the compiler rejects it
if echo 'int main(){return 0;}' | $CC -march=native -x c - -o /dev/null 2>/dev/null; then
  FLAGS="$FLAGS -march=native"
fi
$CC $FLAGS -o toykern3.so toykern3.c -lm
$CC $FLAGS -o genkern.so genkern.c -lm
$CC $FLAGS -o genkern_gi.so genkern_gi.c -lm
echo "built toykern3.so genkern.so genkern_gi.so with: $CC $FLAGS"
