# ------------------------------------------------------------------
# commit: run pre-commit, re-stage any formatter changes, then commit.
#
# Usage:
#   make commit MSG="your commit message"
#
# What it does:
#   1. Run pre-commit on all staged files.
#      ruff may fix imports; ruff-format may reformat.
#   2. git add -u  — re-stage any files the formatters changed.
#   3. Run pre-commit again to confirm it now passes cleanly.
#   4. git commit with your message.
#
# Why the loop: ruff --exit-non-zero-on-fix causes the first run to
# fail when it makes fixes. The second run always passes (nothing left
# to fix). Two runs is always enough.
# ------------------------------------------------------------------

.PHONY: commit
commit:
	@if [ -z "$(MSG)" ]; then echo "Usage: make commit MSG=\"your message\""; exit 1; fi
	@STAGED=$$(git diff --cached --name-only); \
	if [ -z "$$STAGED" ]; then echo "Nothing staged."; exit 1; fi; \
	pre-commit run --files $$STAGED || true; \
	git add -u; \
	pre-commit run --files $$(git diff --cached --name-only); \
	git commit -m "$(MSG)"
