#! /bin/bash

PYTHON_VERSION=3.7.2
VENV_NAME=venv.vibsym

pyenv install $PYTHON_VERSION
pyenv virtualenv $PYTHON_VERSION $VENV_NAME
pyenv local $VENV_NAME
echo "INSTALL requirements.txt"
pip install requirements.txt
echo "INSTALL requirements-dev.txt"
pip install requirements-dev.txt
echo "INSTALL vibsym"
pip install -e .

