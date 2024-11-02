help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (black).
	pre-commit install
	pre-commit run fmt

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && pre-commit run clippy

check:
	pre-commit install && pre-commit run cargo-check

test:
	cargo test --verbose