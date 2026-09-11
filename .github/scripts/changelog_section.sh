#!/usr/bin/env bash
# Print one release section from CHANGELOG.md.
# Usage: bash changelog_section.sh <changelog> <version>
set -euo pipefail

changelog=${1:?missing changelog path}
version=${2:?missing version}

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "version must be X.Y.Z: $version" >&2
  exit 1
}
[[ -f "$changelog" ]] || {
  echo "missing $changelog" >&2
  exit 1
}

header_count=$(awk -v version="$version" '
  BEGIN { pattern = "^##[[:space:]]+(\\[" version "\\]|v?" version ")([[:space:]]|$)" }
  $0 ~ pattern { count++ }
  END { print count + 0 }
' "$changelog")
if [[ "$header_count" -ne 1 ]]; then
  echo "expected one top-level changelog section for $version, found $header_count" >&2
  exit 1
fi

header=$(awk -v version="$version" '
  BEGIN { pattern = "^##[[:space:]]+(\\[" version "\\]|v?" version ")([[:space:]]|$)" }
  $0 ~ pattern { print; exit }
' "$changelog")
body=$(awk -v version="$version" '
  BEGIN { pattern = "^##[[:space:]]+(\\[" version "\\]|v?" version ")([[:space:]]|$)" }
  $0 ~ pattern { found = 1; next }
  found && /^##[[:space:]]/ { exit }
  found { print }
' "$changelog")
body=$(printf '%s\n' "$body" | sed -e '/[^[:space:]]/,$!d')

if [[ "$header" =~ [Uu]nreleased ]] || printf '%s\n' "$body" | grep -Eiq 'unreleased'; then
  echo "changelog section for $version is still marked unreleased" >&2
  exit 1
fi
[[ -n "$body" ]] || {
  echo "empty changelog section for $version" >&2
  exit 1
}

printf '%s\n' "$body"
