.PHONY: clean, run

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	sudo rm -rf outputs/ saved_models/

run:
	python3 server.py
