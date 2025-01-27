# Contributing to SynergyML


## Create virtual environment
```shell
python -m venv synergyml-env


# Activate virtual environment
# On Windows:
synergyml-env\Scripts\activate
# On Unix/MacOS:
source synergyml-env/bin/activate

# Install base package with vision and development tools
pip install "synergyml[vision,dev]"


## Install dependencies

pip install -r requirements-dev.txt


pip install -e .

# Install in development mode with extras
pip install -e ".[vision,dev]"
```
