# Makefile for Stat 159 HW3 – LIGO Gravitational Wave Detection

# Create or update the conda environment
env:
	@echo ">>> Creating or updating the conda environment..."
	conda env update -f environment.yml --prune || conda env create -f environment.yml

# Build the HTML version of the MyST site
html:
	@echo ">>> Building MyST HTML site..."
	myst build --html

# Clean up figures, audio, and build directories
clean:
	@echo ">>> Cleaning up generated files..."
	rm -rf figures/* audio/* _build/*
	@echo "Clean complete."
