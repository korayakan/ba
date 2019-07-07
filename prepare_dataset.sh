#!/bin/bash

[ -d "ba_dataset" ] && echo "Dataset directory exists."
[ ! -d "ba_dataset" ] && echo "Dataset directory DOES NOT exist. Cloning from Github..." && git clone -q https://github.com/korayakan/ba_dataset.git && rm -rf ba_dataset/.git && echo "Done"
