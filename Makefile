.PHONY: clean, run

clean:
	sudo rm -rf outputs/ saved_models/
	find . -type d -name '__pycache__' -exec rm -r {} +

run:
	python3 server.py
