
.PHONY: help
help:
	@echo "Usage:"
	@echo "    make <target>"
	@echo
	@echo "Targets:"
	@echo "    prepare"
	@echo "        Prepare public dataset."
	@echo


.PHONY: prepare
prepare:
	python ./prepare/download_raw.py
	mkdir -p ./src/data
	python ./prepare/split_raw.py
	python ./prepare/prepare_question.py
