#!/bin/bash

echo "Cloning pretrained net from Github..." && rm -rf pretrained_net && git clone -q https://github.com/korayakan/pretrained_net.git && rm -rf ba/.git && echo "Done"
