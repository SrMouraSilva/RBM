#!/bin/bash
sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
python setup.py develop
