#!/bin/bash

cd $( dirname ${BASH_SOURCE[0]} )

FILE_LIST=""
for dir_name in tests tgl/include tgl/src; do
	 FILE_LIST="${FILE_LIST} $( ls $dir_name/*.cpp 2> /dev/null)"
	 FILE_LIST="${FILE_LIST} $( ls $dir_name/*.h 2> /dev/null)"
	 FILE_LIST="${FILE_LIST} $( ls $dir_name/*.cu 2> /dev/null)"
	 FILE_LIST="${FILE_LIST} $( ls $dir_name/*.cuh 2> /dev/null)"
done

for filename in $FILE_LIST; do
    clang-format -style=file -i  $filename
done
