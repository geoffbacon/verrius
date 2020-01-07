SHELL = /bin/bash
SRC = src
DATA = data

# Development-related targets

# list dependencies
requirements: install
	pipreqs --force --savepath requirements.txt $(SRC)

# install dependencies for production
install:
	python -m pip install -r requirements.txt

# remove Python file artifacts
clean:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +
	@find . -name '.ipynb_checkpoints' -exec rm -rf {} +

# check style
lint: format
	pylint --exit-zero --jobs=0 --output-format=colorized $(SRC)
	pycodestyle --show-source $(SRC)
	pydocstyle $(SRC)

# format according to style guide
format:
	black $(SRC)
	isort -rc $(SRC)

# Data-related targets

# preprocess data
preprocess:
	jupyter nbconvert --ExecutePreprocessor.timeout=-1 --execute src/EDAP.ipynb && \
	rm src/EDAP.html &

# delete all trained models
refresh:
	@rm -rf models/
	@mkdir models
	@mkdir models/pos
	@touch models/pos/.gitkeep

.PHONY: requirements install clean lint format preprocess refresh