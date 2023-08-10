.PHONY: clean, run

clean:
	sudo rm -rf outputs/
	find . -type d -name '__pycache__' -exec rm -r {} +

run:
	python3 server.py
