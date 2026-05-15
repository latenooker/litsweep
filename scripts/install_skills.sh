#!/usr/bin/env bash
#
# Install litsweep's Claude + Codex skills via symlink.
#
# Idempotent — re-run after `git pull` if the skill body changed (the
# symlink resolves through to the new content; no copy step needed).
# If the litsweep checkout moves, re-run this script from the new path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

linked=0
for tool in claude codex; do
    skill_src="$REPO_ROOT/skills/$tool/litsweep-deploy"
    skill_dst="$HOME/.${tool}/skills/litsweep-deploy"
    if [[ ! -d "$skill_src" ]]; then
        echo "warning: missing source: $skill_src — skipping" >&2
        continue
    fi
    mkdir -p "$(dirname "$skill_dst")"
    # If dst exists and is NOT a symlink, `ln -sfn` would nest a link
    # inside it rather than replace it (silently broken skill). Remove
    # it only when it is an EMPTY dir (harmless); otherwise print an
    # actionable error and skip this tool. Never rm -rf user content.
    if [[ -e "$skill_dst" && ! -L "$skill_dst" ]]; then
        if [[ -d "$skill_dst" ]] && rmdir "$skill_dst" 2>/dev/null; then
            :  # was an empty dir; removed; fall through to ln
        else
            echo "error: $skill_dst exists and is not a symlink." >&2
            echo "       Move or remove it manually, then re-run." >&2
            continue
        fi
    fi
    ln -sfn "$skill_src" "$skill_dst"
    echo "linked $skill_dst -> $skill_src"
    linked=$((linked + 1))
done

echo
if (( linked == 0 )); then
    echo "error: no skills were linked — see messages above." >&2
    exit 1
fi
echo "done ($linked skill(s) linked). Restart your agent CLI to pick them up."
