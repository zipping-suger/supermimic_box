# supermimic_box

## Delegating implementation to Codex

Claude plans, writes the brief and reviews. Codex does the implementation-heavy,
well-specified work: multi-file edits, boilerplate, mechanical refactors, test
writing. Make small edits (a few lines) directly, because a brief costs more
than the edit.

1. Write the brief to a file in the scratchpad. Codex starts cold, so name the
   files, the constraints, what done looks like and how to check it. Every
   brief also says:
   - run Python as `.venv/bin/python` (tests: `.venv/bin/python -m pytest`);
   - the sandbox has no network, so do not install or download anything;
   - leave `data/`, `logs/` and `results/` alone unless the task is about them.
2. Run `.claude/codex-run.sh <brief-file>` in the background. Codex's default
   here is `xhigh` reasoning, which can take many minutes; for mechanical work
   append `-c 'model_reasoning_effort="medium"'`.
3. Read the summary the script prints, then review
   `.codex-runs/<stamp>/changes.patch`. Open `log.txt` only when a run failed.
4. Verify the result yourself (run the tests). Codex's report is a claim, not
   proof.
5. Send corrections to the same Codex session, which still has its context:
   `.claude/codex-run.sh --resume <session-id> <brief-file>`.

The `codex` plugin (codex-plugin-cc) is also installed. Use it for Codex code
reviews (`/codex:review`, `/codex:adversarial-review`). For implementation work
prefer the script above over the plugin's `codex:codex-rescue` agent, because
the script pins the sandbox to this project and records a patch per run.

Run one Codex task at a time: the patch is a before/after snapshot of the whole
working copy, so concurrent edits (Codex's or Claude's) land in the same patch.
