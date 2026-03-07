#!/usr/bin/env bash
# Setup GitHub branch protection for main using the GitHub REST API via curl.
#
# Requires a GitHub personal access token (classic, with repo scope) or a
# fine-grained token with "Administration" read/write on the repository.
#
# Usage:
#   GITHUB_TOKEN=ghp_... bash scripts/setup_branch_protection.sh
#
# Required status checks match the exact context names reported by
# GitHub Actions: "<workflow_name> / <job_display_name>"
# Workflow name is "CI" (name: field in ci.yml).

set -euo pipefail

REPO="SavaStevanovic/StoreSim"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Error: GITHUB_TOKEN environment variable is not set." >&2
  echo "Usage: GITHUB_TOKEN=ghp_... bash $0" >&2
  exit 1
fi

echo "Configuring branch protection for 'main' on $REPO ..."

curl -fsSL \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/$REPO/branches/main/protection" \
  -d @- <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "CI / Lint & Type Check",
      "CI / Tests (Python 3.10)",
      "CI / Tests (Python 3.11)",
      "CI / Tests (Python 3.12)",
      "CI / pre-commit"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "Done. Branch protection on 'main' updated."
