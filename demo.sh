#!/bin/bash
PORT=5000

# Issue with Pytorch and invalid ELF header, solution based on: https://discuss.pytorch.org/t/importerror-home-name-anaconda3-envs-tf21-lib-python3-7-site-packages-torch-lib-libnvtoolsext-so-1-invalid-elf-header/91764/5
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
echo "Starting Concept Map Recommener at $PORT"
echo "See for documentation: http://localhost:$PORT/"

export FLASK_APP=Webservice.py
export FLASK_ENV=production
flask run --host=0.0.0.0 --no-debugger --port=$PORT
