#!/bin/bash

# optional: start your ssh server on docker start
env > /root/.ssh/environment &
chmod 600 /root/.ssh/environment &
/usr/sbin/sshd -D &
# optional: start your jupyter lab/notebook on docker start
# jupyter lab --no-browser --allow-root --port=8888 --ip=0.0.0.0 --config=/project/docker/jupyter_notebook_config.py
