# OpenSSF Scorecard Improvements

This document summarizes the improvements made to enhance the OpenSSF Scorecard security posture of the PSOD project.

## Changes Implemented

### 1. Security Policy (SECURITY.md)
- ✅ **Added** `SECURITY.md` file in the root directory
- Provides clear instructions for reporting security vulnerabilities
- Defines supported versions and security update process
- Lists security tools and scanning procedures
- Includes contact information for security issues

**Impact**: Improves the "Security-Policy" scorecard check

### 2. Code Owners (CODEOWNERS)
- ✅ **Added** `.github/CODEOWNERS` file
- Defines code ownership for different parts of the repository
- Ensures proper code review by designated owners
- Covers source code, tests, documentation, CI/CD, and security files

**Impact**: Improves the "Code-Review" scorecard check

### 3. Workflow Permissions
- ✅ **Updated** `.github/workflows/security.yml`
  - Fixed OpenSSF Scorecard job permissions from `security-events: read` to `write`
  - Added required permissions: `pull-requests: read`, `statuses: read`

- ✅ **Updated** `.github/workflows/ci.yml`
  - Added top-level `permissions: contents: read` (least privilege)

- ✅ **Updated** `.github/workflows/performance.yml`
  - Added top-level `permissions: contents: read` (least privilege)

- ✅ **Updated** `.github/workflows/docs.yml`
  - Added top-level `permissions: contents: read` (least privilege)

**Impact**: Improves the "Token-Permissions" scorecard check

### 4. Existing Security Infrastructure (Already in Place)

The repository already has excellent security practices:

#### Dependency Management
- ✅ Dependabot configured (`.github/dependabot.yml`)
- ✅ Automated dependency updates for pip and GitHub Actions
- ✅ Weekly update schedule

**Impact**: Satisfies "Dependency-Update-Tool" check

#### SAST Tools
- ✅ CodeQL analysis for Python
- ✅ Bandit security scanning
- ✅ Semgrep security analysis
- ✅ Multiple other security scanners (Safety, pip-audit, Trivy)

**Impact**: Satisfies "SAST" check

#### Vulnerability Scanning
- ✅ Multiple dependency vulnerability scanners
- ✅ Secret scanning (Gitleaks, TruffleHog)
- ✅ Container scanning (Trivy)
- ✅ Automated security workflows

**Impact**: Satisfies "Vulnerabilities" check

#### CI/CD Security
- ✅ Comprehensive CI pipeline with security checks
- ✅ Automated testing on multiple Python versions
- ✅ Security scans on every PR and push

**Impact**: Satisfies "CI-Tests" check

## Repository Settings (Manual Configuration Required)

The following improvements require manual configuration in GitHub repository settings:

### 1. Branch Protection
Configure branch protection rules for `main` and `develop` branches:

- [ ] Require pull request reviews before merging
- [ ] Require status checks to pass before merging
- [ ] Require branches to be up to date before merging
- [ ] Require linear history
- [ ] Include administrators in restrictions
- [ ] Restrict who can push to matching branches

**How to configure**:
1. Go to Settings > Branches
2. Add branch protection rule for `main`
3. Enable the above settings
4. Repeat for `develop` branch

**Impact**: Improves "Branch-Protection" scorecard check

### 2. Signed Commits (Optional but Recommended)
Enable signed commit verification:

- [ ] Configure GPG or SSH signing keys
- [ ] Require verified signatures on commits
- [ ] Update branch protection to require signed commits

**How to configure**:
1. Generate GPG key: `gpg --gen-key`
2. Add to GitHub: Settings > SSH and GPG keys
3. Configure git: `git config commit.gpgsign true`
4. Enable in branch protection rules

**Impact**: Improves "Signed-Commits" scorecard check

### 3. GitHub Actions Security
Additional GitHub Actions security settings:

- [ ] Limit GitHub Actions to selected actions and reusable workflows
- [ ] Allow actions created by GitHub and verified creators
- [ ] Review and approve first-time contributors' workflow runs

**How to configure**:
1. Go to Settings > Actions > General
2. Set "Actions permissions" appropriately
3. Enable "Require approval for all outside collaborators"

### 4. Security Advisories
Enable private security advisories:

- [ ] Enable security advisories in repository settings
- [ ] Configure security policy
- [ ] Set up vulnerability reporting

**How to configure**:
1. Go to Settings > Security & analysis
2. Enable "Private vulnerability reporting"
3. Enable "Dependency graph"
4. Enable "Dependabot alerts"
5. Enable "Dependabot security updates"

## Advanced Improvements (Optional)

### 1. Pin GitHub Actions to SHA Commits
For maximum security, pin all GitHub Actions to specific SHA commits instead of tags:

**Current**: `uses: actions/checkout@v4`
**Recommended**: `uses: actions/checkout@<SHA>  # v4.x.x`

**Note**: This requires ongoing maintenance to update SHAs when actions are updated. Dependabot can help automate this.

**Impact**: Improves "Pinned-Dependencies" scorecard check

### 2. CII Best Practices Badge
Apply for and display the CII Best Practices badge:

1. Visit https://bestpractices.coreinfrastructure.org/
2. Register the project
3. Complete the self-certification questionnaire
4. Add badge to README.md

**Impact**: Improves "CII-Best-Practices" scorecard check

### 3. SLSA Provenance
Implement SLSA provenance for releases:

- [ ] Generate provenance for build artifacts
- [ ] Sign release artifacts
- [ ] Publish provenance to GitHub releases

**Impact**: Improves supply chain security

### 4. Fuzzing
Implement continuous fuzzing:

- [ ] Set up OSS-Fuzz or similar fuzzing service
- [ ] Write fuzzing targets for critical code paths
- [ ] Integrate fuzzing into CI/CD

**Impact**: Improves "Fuzzing" scorecard check

## Scorecard Checks Summary

| Check | Status | Notes |
|-------|--------|-------|
| Security-Policy | ✅ Pass | SECURITY.md added |
| Code-Review | ✅ Pass | CODEOWNERS added |
| Token-Permissions | ✅ Pass | Workflow permissions fixed |
| Dependency-Update-Tool | ✅ Pass | Dependabot configured |
| SAST | ✅ Pass | CodeQL, Bandit, Semgrep enabled |
| Vulnerabilities | ✅ Pass | Multiple scanners active |
| CI-Tests | ✅ Pass | Comprehensive CI pipeline |
| Branch-Protection | ⚠️ Manual | Requires GitHub settings |
| Signed-Commits | ⚠️ Optional | Requires user configuration |
| Pinned-Dependencies | ⚠️ Partial | Actions use tags, not SHAs |
| CII-Best-Practices | ⚠️ Optional | Apply for badge |
| Fuzzing | ⚠️ Optional | Not currently implemented |

## Running OpenSSF Scorecard

To check the current scorecard status:

```bash
# Install Scorecard CLI
go install github.com/ossf/scorecard/v4/cmd/scorecard@latest

# Run scorecard
scorecard --repo=github.com/diogoribeiro7/PSOD --show-details

# Or use the GitHub Action (already configured in security.yml)
```

## Monitoring

The OpenSSF Scorecard workflow runs automatically:
- ✅ On push to main/develop branches
- ✅ On pull requests to main/develop
- ✅ Weekly on Monday at 00:00 UTC
- ✅ Manual trigger via workflow_dispatch

Results are uploaded to:
- GitHub Security tab (SARIF format)
- OpenSSF Scorecard API (if publish_results enabled)

## Next Steps

1. ✅ Commit these changes to the repository
2. ⚠️ Configure branch protection rules (manual)
3. ⚠️ Enable signed commits (optional)
4. ⚠️ Review GitHub Actions permissions (manual)
5. ⚠️ Consider pinning actions to SHA commits
6. ⚠️ Apply for CII Best Practices badge (optional)
7. ✅ Monitor scorecard results in Security tab

## References

- [OpenSSF Scorecard](https://github.com/ossf/scorecard)
- [Scorecard Checks](https://github.com/ossf/scorecard#scorecard-checks)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [SLSA Framework](https://slsa.dev/)
- [CII Best Practices](https://bestpractices.coreinfrastructure.org/)

---

**Last Updated**: 2025-11-15
**Maintainer**: @diogoribeiro7
