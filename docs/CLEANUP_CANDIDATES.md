# Cleanup Candidates

Use this inventory before deleting old files:

```bash
.venv_integrated/bin/python tools/inventory_cleanup_candidates.py
```

The output is intentionally conservative:

- `compatibility_wrapper`: old user-facing entrypoints. Delete only after
  confirming shortcuts, docs, and automation no longer reference them.
- `local_generated_state`: local state that should be cleaned with
  `tools/clean_generated.py`, not by hand.
- `large_source_file`: files that should be split or isolated over time, not
  deleted directly.

Deletion remains a confirmation step. The inventory exists so review can be
objective instead of relying on memory.

