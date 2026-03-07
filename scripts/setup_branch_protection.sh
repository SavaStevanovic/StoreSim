#!/usr/bin/env bash
# Setup GitHub branch protection for main.
# Requires: gh CLI authenticated with admin access to the repo.
#
# Required status checks (must match job names in .github/workflows/ci.yml):
#   - "Lint & Type Check"
#   - "Tests (Python 3.10)" / "Tests (Python 3.11)" / "Tests (Python 3.12)"
#   - "pre-commit"
#
# Run once after pushing the CI workflow to GitHub.

set -euo pipefail

REPO="SavaStevanovic/StoreSim"

echo "Configuring branch protection for 'main' on $REPO ..."

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/$REPO/branches/main/protection" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Lint & Type Check",
      "Tests (Python 3.10)",
      "Tests (Python 3.11)",
      "Tests (Python 3.12)",
      "pre-commit"
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

echo "Branch protection configured. Direct pushes to main are now blocked."
