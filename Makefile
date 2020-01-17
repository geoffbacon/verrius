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

# format according to style guide
format:
	black $(SRC)
	isort -rc $(SRC)

# check style
lint: format
	pylint --exit-zero --jobs=0 --output-format=colorized $(SRC)
	pycodestyle --show-source $(SRC)
	pydocstyle $(SRC)

# Data-related targets

# create orthography profile
profile:
	python src/orthprofile.py

# preprocess data
preprocess:
	python src/preprocessing.py prepare_pos

# create unlabelled corpus
corpus:
	python src/corpus.py

# train character and word embeddings
embeddings:
	nohup python -u src/embeddings.py > /dev/null &

# delete all trained models
refresh:
	@rm -rf models/
	@mkdir models models/pos models/embeddings
	@touch models/pos/.gitkeep models/embeddings.gitkeep

.PHONY: requirements install clean lint format profile preprocess corpus embeddings refresh