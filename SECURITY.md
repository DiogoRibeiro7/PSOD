# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of PSOD seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred)
   - Navigate to the [Security tab](https://github.com/diogoribeiro7/PSOD/security/advisories/new)
   - Click "Report a vulnerability"
   - Fill out the form with details about the vulnerability

2. **Email**
   - Send an email to: dfr@esmad.ipp.pt
   - Include "[SECURITY]" in the subject line
   - Provide detailed information about the vulnerability

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### What to Expect

You can expect to receive a response within:
- **48 hours** - Initial acknowledgment of your report
- **5 business days** - Detailed response with next steps
- **30 days** - Resolution or status update on the vulnerability

We will:
1. Acknowledge receipt of your vulnerability report
2. Assess the vulnerability and determine its impact
3. Work on a fix and coordinate the release
4. Credit you for the discovery (unless you prefer to remain anonymous)

### Security Update Process

When a security vulnerability is identified:

1. A fix is developed and tested
2. A security advisory is published on GitHub
3. A new version is released with the fix
4. The vulnerability is disclosed publicly after users have had time to update

### Security Best Practices

When using PSOD:

- Always use the latest version
- Keep all dependencies up to date
- Follow secure coding practices when integrating PSOD
- Validate and sanitize all input data
- Use appropriate error handling and logging
- Review security advisories regularly

### Scope

The following are **in scope** for security reports:

- Code execution vulnerabilities
- Authentication/authorization bypass
- Data injection vulnerabilities
- Information disclosure
- Cryptographic issues
- Dependency vulnerabilities that affect PSOD

The following are **out of scope**:

- Social engineering attacks
- Physical attacks
- Denial of Service (DoS) attacks
- Issues in dependencies that don't affect PSOD
- Issues that require physical access to a user's device

### Disclosure Policy

- We ask that you give us reasonable time to fix the issue before public disclosure
- We will credit you for finding the vulnerability (unless you prefer anonymity)
- We will work with you to understand and address the issue

### Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue.

## Security Tools and Scanning

This project uses multiple security scanning tools:

- **CodeQL** - Static analysis for security vulnerabilities
- **Bandit** - Python security linting
- **Safety** - Dependency vulnerability scanning
- **pip-audit** - Dependency auditing
- **Semgrep** - Static analysis security scanning
- **Trivy** - Comprehensive vulnerability scanner
- **Gitleaks** - Secret scanning
- **TruffleHog** - Secret detection
- **OpenSSF Scorecard** - Security posture assessment

## Security Hardening

We implement several security hardening measures:

- Automated security scanning on all pull requests
- Regular dependency updates via Dependabot
- Code review requirements for all changes
- Signed commits (recommended)
- Branch protection on main branches
- Least privilege access controls

## Third-Party Security

PSOD depends on several third-party libraries. We:

- Monitor security advisories for all dependencies
- Update dependencies promptly when security issues are discovered
- Use automated tools to detect vulnerable dependencies
- Maintain an up-to-date list of dependencies

## Contact

For security-related questions or concerns, contact:
- Email: dfr@esmad.ipp.pt
- GitHub: @diogoribeiro7

## Acknowledgments

We would like to thank the following security researchers and contributors who have helped improve the security of PSOD:

(This section will be updated as we receive security reports)

---

Thank you for helping keep PSOD and our users safe!

