# GitHub Actions major-version bump audit

Dependabot's `github-actions` ecosystem (added in #146) merged several major-version
jumps purely on "CI stayed green": `actions/checkout` v4→v7, `actions/setup-python`
v5→v7, `actions/setup-node` v4→v7, `actions/github-script` v8→v9. #167 asked for a
one-time read of each action's release notes to confirm nothing behavior-relevant
changed silently. Findings below - re-run this check the next time any of these
jump a major version.

`github/codeql-action` (bumped v3 → v4 in `codeql.yml`, see below) is also
Dependabot-managed and in scope for this same habit (#188).

## `actions/checkout` (v4 → v7)

**Real change:** v4.4.0+/v7.0.0 added a safety block for checking out fork PRs
under `pull_request_target` / `workflow_run` triggers (an `allow-unsafe-pr-checkout`
override was added, marked `[BREAKING]`).

**Applicability:** `.github/workflows/first.interaction.yml` uses `pull_request_target`
but never calls `actions/checkout` - it only uses `actions/github-script`.
`.github/workflows/pr-review-ping.yml` also triggers on `pull_request_target` and
does call `actions/checkout@v7`, but its step doesn't override `ref:`, so it checks
out the base branch rather than the fork PR's head - the exact scenario the
`[BREAKING]` safety block guards against never arises here either. **Not applicable.**

## `actions/setup-python` (v5 → v7)

**Real changes:** v6.0.0 requires Node 24 (runner ≥ v2.327.1); v7.0.0 removed the
`pip-install` input.

**Applicability:** every `setup-python` step in this repo only sets `python-version`
and, in `audits.yml`, `cache: 'pip'` - neither removed nor Node-runtime-sensitive.
**Not applicable.**

## `actions/setup-node` (v4 → v7)

**Real changes:** v5.0.0 auto-enables caching when `package.json` has a
`packageManager` field (opt out via `package-manager-cache: false`); v6.0.0 scoped
automatic caching to npm only; v7.0.0 removed a dummy `NODE_AUTH_TOKEN` export.

**Applicability:** this repo has no `package.json` at all (no npm dependency
management), so there's nothing for the `packageManager` auto-caching to detect,
and nothing depends on `NODE_AUTH_TOKEN`. **Not applicable.**

## `actions/github-script` (v8 → v9)

**Real changes:** `@actions/github` is now ESM-only, so `require('@actions/github')`
no longer works; scripts that declare `const getOctokit = ...`/`let getOctokit = ...`
now collide with the injected parameter of the same name.

**Applicability:** `first.interaction.yml`'s script only uses the injected
`github`/`context` globals directly - no `require()`, no `getOctokit` redeclaration.
**Not applicable.**

## `github/codeql-action` (v3 → v4)

**Real changes:** runs on the Node 24 runtime instead of Node 20; bumped the
minimum CodeQL bundle version to 2.19.4; removed the `add-snippets` input on
`analyze` (deprecated since 3.26.4); deprecated the undocumented
`CODEQL_ACTION_CLEANUP_TRAP_CACHES` env var; starting April 2026, file coverage
collection is skipped on pull-request analyses (still computed on push/schedule
runs). v3 itself is now deprecated, scheduled for removal alongside GHES 3.19 in
December 2026.

**Applicability:** `.github/workflows/codeql.yml`'s three steps
(`init`/`autobuild`/`analyze`) only set `languages` and `config-file` - no
`add-snippets`, no `CODEQL_ACTION_CLEANUP_TRAP_CACHES` anywhere in this repo.
`ubuntu-latest` already runs Node 24 (needed for `actions/setup-python` v6+
too, audited above), so the runtime bump is a non-issue. The skipped
PR-only file coverage doesn't affect analysis results, only an informational
UI panel. **Not applicable** - this bump was safe, though it landed via a
direct commit (`b632cfb`) rather than going through this audit first, exactly
the gap #188 asked to close. Treat this entry as the audit that should have
gated that merge, not one that happened before it.

## Job timeouts and concurrency (#1050)

GitHub's default job timeout is six hours, so a hung benchmark or browser job can hold a
runner for that long, and pushing again to a pull request used to leave every superseded
run going. Policy for every workflow in `.github/workflows/`:

- **Every job sets `timeout-minutes`.** Recent runs finish in under five minutes end to end,
  so the limits leave a wide margin: 5 for the small check jobs, 10 for lint, favicons,
  and explainers, 15 for the audit matrix, 20 for the profiler tests, 30 for
  the benchmark harness and CodeQL. Raise a limit in the same PR that makes a job slower.
- **Workflows that run on `pull_request` set `concurrency`**, grouped by workflow and pull
  request number (or ref for pushes). `cancel-in-progress` is on only for `pull_request`
  events, so a new commit cancels the stale PR run but a run on `main` is never cancelled.
- **Left alone:** `first.interaction.yml` and `pr-review-ping.yml` (`pull_request_target` and
  `issues` events post one-off comments and must not be cancelled).
- **Not changed:** several workflows still trigger on both `push` and `pull_request`, so a
  commit on a branch with an open PR starts each of those jobs twice. Restricting `push` to
  `main` would fix that but changes which events gate a merge, so it is a separate decision.

## Conclusion

All five audited version jumps (including `github/codeql-action`'s first
major bump since it was added to Dependabot's ecosystem) are confirmed safe
for how this repo actually uses them. None of the breaking changes in any
release apply to the specific inputs/triggers configured here.

**Maintenance:** If Dependabot (or a manual update) proposes a future **major
version bump** for any audited GitHub Action, repeat this audit and update this
document with the findings for the new version(s).
