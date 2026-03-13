# Security Policy

## Supported Versions

AutoLabel is an active beta project. Security fixes are expected to land on the latest released code first.

| Version | Supported |
|---------|-----------|
| 1.0.x | Yes |
| < 1.0.0 | No |

## Reporting a Vulnerability

Please avoid posting full exploit details in a public issue.

Preferred path:

1. Use GitHub Security Advisories if private reporting is enabled for the repository.
2. If private reporting is not available, open a minimal public issue without exploit details and request a private coordination channel with the maintainer.

When reporting, include:

- affected file or subsystem
- reproduction steps
- impact assessment
- whether the issue can lead to code execution, secret exposure, data corruption, or sandbox escape

## Scope Notes

AutoLabel executes LLM-generated labeling functions only after AST validation and sandbox checks, but treat any code-generation pathway as security-sensitive. Reports involving:

- sandbox bypasses
- unsafe AST validation gaps
- secret leakage
- provider credential handling
- unsafe file or network access

are especially valuable.
