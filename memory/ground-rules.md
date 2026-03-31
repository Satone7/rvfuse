<!--
  Sync Impact Report
  ===================
  Version change: N/A → 1.0.0
  Added sections:
    - Core Principles (4 principles)
    - Development Workflow
    - Quality Gates
    - Governance
  Removed sections: N/A (initial creation)
  Templates requiring updates:
    - ✅ design-template.md: Ground-rules Check section aligned (no changes needed)
    - ✅ spec-template.md: Requirements section aligned (no changes needed)
    - ✅ tasks-template.md: Task categorization aligned (no changes needed)
  Follow-up TODOs: None
-->

# RVFuse Ground-rules

## Core Principles

### I. Code Quality

Code MUST be readable, maintainable, and follow consistent conventions across the project.

- All code MUST use consistent naming conventions (camelCase for functions/variables, PascalCase for types/classes)
- Functions MUST have single, clear purposes; maximum 50 lines unless complexity is justified
- Code MUST be self-documenting; comments reserved for non-obvious logic or business rules
- Dependencies MUST be explicit; no hidden global state or implicit imports
- Error handling MUST be explicit; exceptions caught and handled at appropriate levels
- Magic numbers and hardcoded values MUST be replaced with named constants or configuration
- Duplication MUST be eliminated; three similar code blocks trigger refactoring

**Rationale**: Readable code reduces cognitive load, accelerates onboarding, and minimizes bugs.
Long-term maintainability depends on consistent patterns that any developer can follow.

**How to apply**: Before merging, verify naming consistency, function length, and
explicit error handling. Reject code that relies on implicit behavior.

### II. Testing

All features MUST have automated tests; test coverage MUST NOT decrease on any merge.

- Unit tests MUST cover all public functions with edge cases
- Integration tests MUST verify component interactions and data flows
- Contract tests MUST validate API boundaries and external interfaces
- Tests MUST be deterministic: no flaky tests, no reliance on external state
- Test names MUST describe the scenario and expected outcome
- Broken tests MUST block merges; no skipping or disabling tests
- Test coverage MUST meet minimum threshold (80% for new code)
- Performance tests MUST be included for critical paths

**Rationale**: Tests are the safety net that enables confident refactoring and deployment.
Untested code is legacy code from day one.

**How to apply**: Every PR MUST include tests. Run full test suite before merge.
Flag any coverage decrease as a violation.

### III. User Experience

Features MUST be designed with user needs as the primary focus; usability MUST NOT be sacrificed for developer convenience.

- Interfaces MUST be intuitive; users MUST accomplish primary tasks without documentation
- Error messages MUST be actionable; explain what went wrong and how to fix it
- Response times MUST meet user expectations (100ms for interactions, 1s for operations)
- Accessibility MUST be considered; keyboard navigation, screen reader support, color contrast
- Mobile and desktop experiences MUST be optimized for their respective contexts
- User feedback MUST be collected and incorporated into design decisions
- Loading states MUST provide progress indication for operations >200ms
- Features MUST have graceful degradation for edge cases and errors

**Rationale**: User experience directly impacts adoption, satisfaction, and support costs.
Developer convenience never trumps user needs.

**How to apply**: Validate UI changes with real user scenarios. Test error messages
for actionability. Verify accessibility compliance.

### IV. Performance

Systems MUST meet defined performance targets; performance MUST NOT degrade without justification.

- Response time MUST meet targets: p95 <200ms for reads, <500ms for writes
- Resource usage MUST be bounded: memory, CPU, and I/O within defined limits
- Database queries MUST be optimized; N+1 queries prohibited
- Caching MUST be implemented for frequently accessed data
- Batch operations MUST handle large datasets without timeout or memory overflow
- Performance regressions MUST be detected and fixed before merge
- Performance tests MUST be run for all critical paths on each release
- Monitoring MUST track key metrics (latency, throughput, error rates)

**Rationale**: Performance directly affects user experience and operational costs.
Degradation is often gradual and hard to reverse once entrenched.

**How to apply**: Define performance targets in specs. Run benchmarks on critical paths.
Alert on regressions and fix before deployment.

## Development Workflow

All development MUST follow a structured workflow that ensures quality and traceability.

### Branch Strategy

- Feature branches MUST be created from main with descriptive names (e.g., `123-add-auth`)
- Branches MUST be short-lived (<1 week); large features MUST be broken into smaller PRs
- Main branch MUST always be deployable

### Code Review

- All changes MUST be reviewed by at least one other developer
- Reviews MUST check: correctness, test coverage, performance impact, principle compliance
- Reviewers MUST provide actionable feedback; "looks good" is insufficient
- Authors MUST address all review comments before merge

### Merge Requirements

- All tests MUST pass
- No decrease in test coverage
- No unresolved review comments
- Performance benchmarks MUST pass for critical paths
- Documentation MUST be updated for user-facing changes

## Quality Gates

Quality gates MUST be enforced at each stage of development.

### Pre-Commit

- Linting MUST pass (no warnings)
- Unit tests MUST pass for modified files
- Commit messages MUST follow convention: `type(scope): description`

### Pre-Merge

- Full test suite MUST pass
- Coverage MUST meet threshold (80% new, 70% overall)
- Performance benchmarks MUST pass
- Security scans MUST pass (no new vulnerabilities)

### Pre-Deploy

- Integration tests MUST pass in staging
- Smoke tests MUST pass in production-like environment
- Rollback plan MUST be documented

## Governance

Ground-rules MUST be followed by all contributors; exceptions MUST be documented and approved.

### Amendment Procedure

- Amendments MUST be proposed via pull request to `memory/ground-rules.md`
- Amendments MUST include rationale and impact analysis
- Amendments MUST be approved by project stakeholders
- Version MUST increment according to semantic versioning

### Version Policy

- **MAJOR**: Principle removals or backward-incompatible redefinitions
- **MINOR**: New principles or materially expanded guidance
- **PATCH**: Clarifications, wording fixes, non-semantic refinements

### Compliance Review

- All PRs MUST verify compliance with ground-rules
- Violations MUST be flagged in review comments
- Complexity MUST be justified in design documents when principles are exceeded
- Persistent violations MUST trigger escalation to project stakeholders

**Version**: 1.0.0 | **Ratified**: 2026-03-31 | **Last Amended**: 2026-03-31