#!/bin/bash

# Hardened variant of upstream scripts/install_codex.sh for the E2B harness.
#
# Many benchmark project images pin third-party apt repositories (Kitware,
# Bazel via storage.googleapis.com, LLVM, ...) for older Ubuntu/Debian releases
# that no longer publish a Release file. Upstream's bare `apt-get update` runs
# under `bash -eux`, so a single dead repo aborts the whole agent-tooling
# install and the task never runs (benchmark status "error").
#
# The agent tooling only needs curl (from the base distro) plus Node via nvm,
# so we disable every non-base apt source before updating. This runs only in
# the agent container; the project's own build and the separate nested
# validation containers used for stages S1-S4 are unaffected.

set -eux

if ! command -v curl >/dev/null; then
  # Move any third-party apt source aside, keeping only the base Ubuntu/Debian
  # distro repositories (which provide curl).
  disabled=/etc/apt/sources.list.d.disabled
  mkdir -p "$disabled"
  for f in /etc/apt/sources.list.d/*; do
    [ -e "$f" ] || continue
    if grep -qiE '(archive|security|ports|old-releases)\.ubuntu\.com|deb\.debian\.org|security\.debian\.org|deb\.security\.debian\.org' "$f"; then
      continue
    fi
    mv "$f" "$disabled"/ 2>/dev/null || true
  done

  # Strip known-dead third-party repos if they were added to the classic single
  # sources.list file instead of sources.list.d.
  if [ -f /etc/apt/sources.list ]; then
    sed -ri '/apt\.kitware\.com|bazel-apt|apt\.llvm\.org|packages\.cloud\.google\.com/d' /etc/apt/sources.list 2>/dev/null || true
  fi

  # First attempt with the sanitized source set; if a base repo still fails to
  # update (e.g. an EOL release moved to old-releases), fall back to disabling
  # every extra source and retry so curl can still be installed.
  if ! apt-get update; then
    mv /etc/apt/sources.list.d/* "$disabled"/ 2>/dev/null || true
    apt-get update
  fi

  apt-get install -y curl
fi

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash

source "$HOME/.nvm/nvm.sh"

nvm install 22
npm -v

npm install -g @openai/codex@0.118.0

mkdir -p "$HOME/.codex"
