#!/bin/bash

PREFIX=$( dirname ${BASH_SOURCE[0]} )

cd  ${PREFIX}
rm -rf build
mkdir -p build
cd build
cmake ..
cmake --build .

./tests/test_runner