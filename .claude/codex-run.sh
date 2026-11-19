#!/usr/bin/env bash
# Run one Codex task non-interactively and leave a reviewable record of it.
#
#   .claude/codex-run.sh <brief-file> [extra `codex exec` args...]
#   .claude/codex-run.sh --resume <session-id> <brief-file> [extra args...]
#
# Codex may edit files inside this project only (workspace-write sandbox, no
# network). Each run gets a directory .codex-runs/<stamp>/ holding:
#   brief.md         the brief that was sent
#   log.txt          Codex's full transcript (long; read only when debugging)
#   last-message.md  Codex's final report
#   changes.patch    everything the run changed in the working copy
#   summary.txt      what this script prints: exit code, session id, diffstat,
#                    final report
# The patch comes from two throwaway snapshots, so the real git index, the
# branch and any commits are left alone.
set -uo pipefail

usage() { sed -n '2,6p' "${BASH_SOURCE[0]}" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 2

resume_id=""
if [[ "${1:-}" == "--resume" ]]; then
  [[ $# -ge 3 ]] || usage
  resume_id=$2
  shift 2
fi
[[ $# -ge 1 ]] || usage
brief=$1
shift
[[ -f "$brief" ]] || { echo "brief file not found: $brief" >&2; exit 2; }

run="$ROOT/.codex-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run"
cp "$brief" "$run/brief.md"

# Tree object for the whole working copy (respects .gitignore), built in a
# temporary index so the user's staging area is untouched.
snapshot() {
  local idx="$run/index.$1" tree
  GIT_INDEX_FILE="$idx" git add -A >/dev/null 2>&1
  tree=$(GIT_INDEX_FILE="$idx" git write-tree)
  rm -f "$idx"
  printf '%s\n' "$tree"
}

before=$(snapshot before) || { echo "snapshot failed (is this a git repo?)" >&2; exit 2; }

if [[ -n "$resume_id" ]]; then
  cmd=(codex exec resume -c 'sandbox_mode="workspace-write"')
  target=("$resume_id")
else
  cmd=(codex exec -s workspace-write -C "$ROOT")
  target=()
fi

"${cmd[@]}" --skip-git-repo-check -o "$run/last-message.md" "$@" \
  "${target[@]}" - <"$run/brief.md" >"$run/log.txt" 2>&1
status=$?

after=$(snapshot after)
git diff --binary "$before" "$after" >"$run/changes.patch"
session=$(grep -oiE 'session id: *[0-9a-f-]{20,}' "$run/log.txt" | head -1 | grep -oE '[0-9a-f-]{20,}$')

{
  echo "run dir:    ${run#"$ROOT"/}"
  echo "exit code:  $status"
  echo "session id: ${session:-unknown}"
  echo
  echo "== files changed =="
  git diff --stat "$before" "$after"
  [[ -s "$run/changes.patch" ]] || echo "(none)"
  echo
  echo "== codex final message =="
  if [[ -s "$run/last-message.md" ]]; then
    cat "$run/last-message.md"
    echo
  else
    echo "(no final message; last lines of log.txt follow)"
    tail -n 30 "$run/log.txt"
  fi
} | tee "$run/summary.txt"

exit "$status"
