#!/usr/bin/env bash
#
# Install litsweep's Claude + Codex skills via symlink.
#
# Idempotent — re-run after `git pull` if the skill body changed (the
# symlink resolves through to the new content; no copy step needed).
# If the litsweep checkout moves, re-run this script from the new path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for tool in claude codex; do
    skill_src="$REPO_ROOT/skills/$tool/litsweep-deploy"
    skill_dst="$HOME/.${tool}/skills/litsweep-deploy"
    if [[ ! -d "$skill_src" ]]; then
        echo "missing source: $skill_src" >&2
        continue
    fi
    mkdir -p "$(dirname "$skill_dst")"
    ln -sfn "$skill_src" "$skill_dst"
    echo "linked $skill_dst -> $skill_src"
done

echo
echo "done. Restart your agent CLI to pick up new skills."
