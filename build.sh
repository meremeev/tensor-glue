#!/bin/bash

PREFIX=$( dirname ${BASH_SOURCE[0]} )

cd  ${PREFIX}
rm -rf build
mkdir -p build
cd build
cmake ..
cmake --build .

printf "\n-- Platform info\n"
./tests/platform_info

printf "\n-- Header-only library test\n"
./tests/header_only_lib_test

printf "\n-- Binary library test\n"
./tests/binary_lib_test