#!/bin/bash
set -e

# curl -sSL https://install.python-poetry.org | python3 -
# echo 'export PATH="/home/jovyan/.local/bin:$PATH"' >> ~/.bashrc
# source ~/.bashrc

# mkdir diplom
# cd diplom

# git clone https://github.com/kezouke/ReProcess.git
# git clone https://github.com/KonstFed/RAGC.git
# git clone https://github.com/Andrchest/SemanticGraphParser.git

# cd SemanticGraphParser
# git switch costil-posix
# cd ..

# cd RAGC
# git config user.email "konstantin_fedorov_03@mail.ru"
# git config user.name "KonstFed"
poetry config virtualenvs.in-project true
rm poetry.lock
poetry install

curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve &
