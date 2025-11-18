.PHONY: ignore-git-repos

ignore-git-repos:
	@if [ ! -f .gitignore ]; then touch .gitignore; fi
	@echo "# Auto-ignored nested Git repositories - $(shell date)" >> .gitignore
	find . -name ".git" -type d ! -path "./.git/*" | \
	sed 's/\/\.git$$//' | sed 's/^\.\///' | \
	grep -v '^\.*$$' | sort | uniq >> .gitignore
