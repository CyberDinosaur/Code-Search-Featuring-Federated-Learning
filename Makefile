.PHONY: clean, run

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -type d -name '.ipynb_checkpoints' -exec rm -r {} +
	sudo rm -rf outputs/ saved_models/

run:
	python3 server.py
