
.PHONY: help
help:
	@echo "Usage:"
	@echo "    make <target>"
	@echo
	@echo "Targets:"
	@echo "    prepare : Prepare public dataset."
	@echo "    check : Check lint and tests."

.PHONY: prepare
prepare:
	python ./prepare/download_raw.py
	mkdir -p ./data
	python ./prepare/split_raw.py
	python ./prepare/prepare_question.py

.PHONY: check
check:
	pytest tests/
	flake8 src/ tests/
