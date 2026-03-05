#!/usr/bin/env bash
# Open a Terminal session with THIS folder as the working directory.

cd "$(dirname "$0")" || exit 1

# Start an interactive login shell (zsh on modern macOS; falls back to bash)
if command -v zsh >/dev/null 2>&1; then
  exec /bin/zsh -l
else
  exec /bin/bash -l
fi
