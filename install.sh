#!/bin/sh
# conda create --name tf-mnist python=2.7
conda create --name tf-mnist --yes
conda run --name tf-mnist pip install -r requirements.txt
