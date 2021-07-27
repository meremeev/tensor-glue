#!/bin/bash

usage() {
  echo "  Script to build tensor-glue library and tests.
  usage: build.sh [-h | -c | -t ]
    options:
    -h: show usage help
    -c: clean build - remove all artifacts first
    -t: run tests after bulding    
  "
}

REPO_ROOT=$( dirname ${BASH_SOURCE[0]} )
CLEAN_BUILD="no"
RUN_TESTS="no"

while getopts 'hct' OPTION; do
  case "$OPTION" in
    h)
      usage
      exit 0
      ;;
    c)
      CLEAN_BUILD="yes"
      ;;
    t)
      RUN_TESTS="yes"
      ;;
  esac
done

cd  ${REPO_ROOT}

if [ "$CLEAN_BUILD" == "yes" ]; then
  	rm -rf build
	mkdir -p build
	cd build
	cmake ..
else
	cd build
	if [ "$RUN_TESTS" == "yes" ]; then
		rm -f ./tests/platform_info
		rm -f ./tests/header_only_lib_test
		rm -f ./tests/binary_lib_test
	fi
fi


cmake --build .


if [ "$RUN_TESTS" == "yes" ]; then
	printf "\n-- Platform info\n"
	./tests/platform_info

	printf "\n-- Header-only library test\n"
	./tests/header_only_lib_test

	printf "\n-- Binary library test\n"
	./tests/binary_lib_test
fi