#!/bin/sh
for file in $*
do
	conda run --name tf-mnist python predict_interface_usage.py "$file"
done
