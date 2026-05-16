#!/usr/bin/env bash
# Scan git-modified files for common secret patterns.
# Runs after every Edit/Write via PostToolUse hook.

PATTERNS=(
  "sk-[a-zA-Z0-9]{20,}"
  "ANTHROPIC_API_KEY[[:space:]]*="
  "ghp_[a-zA-Z0-9]{36}"
  "api_key[[:space:]]*=[[:space:]]*['\"][^'\"]{8,}"
)

EXIT=0
for f in $(git diff --name-only 2>/dev/null); do
  [ -f "$f" ] || continue
  for pat in "${PATTERNS[@]}"; do
    if grep -qE "$pat" "$f" 2>/dev/null; then
      echo "HOOK WARNING: possible secret in $f (pattern: $pat)" >&2
      EXIT=1
    fi
  done
done
exit $EXIT
