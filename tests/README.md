# Test Layout

- `shared/`: framework-agnostic tests for schema, data infra, processors,
  validators, data views, losses, and project structure.
- `frameworks/<name>/`: framework-specific compatibility tests. Kai0-specific
  launcher, profile, and shell wrapper tests live in `frameworks/kai0/`.

Keep new VLA, 3D, or world-model framework tests under their own
`frameworks/<name>/` directory. Shared contracts should stay in `shared/`.
