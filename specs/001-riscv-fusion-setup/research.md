# Research: RVFuse Project Setup Foundation

**Date**: 2026-03-31 | **Phase**: 0

## Research Tasks

### 1. Dependency Source Verification

**Question**: Are Xuantie repository URLs valid and accessible?

**Method**: Direct URL verification via GitHub API

**Finding**: All three upstream repositories exist and are public:
- Xuantie QEMU: https://github.com/XUANTIE-RV/qemu ✓
- Xuantie LLVM: https://github.com/XUANTIE-RV/llvm-project ✓
- Xuantie newlib: https://github.com/XUANTIE-RV/newlib ✓

**Decision**: Use documented URLs as canonical source references

**Rationale**: Direct GitHub links provide:
- Version tracking via commit history
- Attribution to upstream maintainers
- No authentication required (public repos)

**Alternatives Considered**:
- Mirror repositories: Rejected - would lose upstream attribution
- Package managers: Rejected - Xuantie toolchain not in standard packages
- Vendored copies: Rejected - would lose version tracking

**Architecture Alignment**: ADR-001 mandates git submodules for versionable dependencies

---

### 2. Git Submodule Integration Best Practices

**Question**: What are the recommended patterns for managing large submodules?

**Method**: Git documentation and community practices review

**Finding**: Recommended patterns:
1. Document submodule initialization in setup guide
2. Provide shallow clone option for faster initial setup
3. Pin to specific commits rather than tracking HEAD
4. Document fallback for network failures

**Decision**: Implement documented submodule workflow with:
- Explicit `git submodule add` commands in setup guide
- Shallow clone recommendation (`--depth 1`)
- Commit pinning guidance
- Network failure recovery steps

**Rationale**: Large repositories (QEMU, LLVM) require careful handling to avoid slow setup times

**Alternatives Considered**:
- Automated submodule sync script: Deferred - manual control preferred for first phase
- Containerized dependencies: Deferred - adds complexity for setup phase

**Architecture Alignment**: ADR-001 consequences note slow synchronization - shallow clone mitigates

---

### 3. Documentation Structure Standards

**Question**: What documentation structure supports contributor onboarding?

**Method**: Open-source project documentation patterns (GitHub template projects)

**Finding**: Effective onboarding documentation requires:
1. Clear scope statement (what is/isn't included)
2. Step-by-step setup instructions
3. Verification checklist for completion
4. Dependency table with status (mandatory/optional)

**Decision**: Structure documentation with:
- Architecture.md: Scope, ADRs, quality targets, deferred work
- Setup guide: Prerequisites, steps, verification, fallbacks
- Spec: User stories, requirements, success criteria

**Rationale**: Separation of concerns:
- Architecture for technical decisions
- Setup guide for contributor actions
- Spec for feature requirements

**Alternatives Considered**:
- Single README: Rejected - would mix concerns
- Wiki-style docs: Rejected - harder to version control

**Architecture Alignment**: Quality attribute 6.1 (Setup Clarity) requires explicit current/deferred separation

---

### 4. Optional Dependency Policy Design

**Question**: How to document optional dependencies without ambiguity?

**Method**: Dependency management patterns from similar projects

**Finding**: Clear optional dependency documentation requires:
1. Explicit "optional" label
2. Activation condition documented
3. Impact of missing dependency stated
4. Source reference preserved even when optional

**Decision**: Document newlib with:
- Label: "Optional in current phase"
- Activation: "Required when bare-metal runtime support needed"
- Impact: "Setup completable without newlib"
- Source: Preserved canonical URL

**Rationale**: ADR-003 mandates optional status - documentation must reflect this clearly

**Alternatives Considered**:
- Omit optional dependencies: Rejected - violates ADR-003 source preservation
- Mark as "future required": Rejected - introduces ambiguity

**Architecture Alignment**: Quality attribute 6.2 (Reproducibility) requires optional dependency conditions

---

## Resolved Clarifications

All technical context items resolved - no NEEDS CLARIFICATION markers remain.

| Item | Resolution |
|------|------------|
| Language/Version | N/A (documentation phase) |
| Primary Dependencies | Git, Markdown |
| Testing | Documentation review |
| Performance Goals | 30-min setup (per architecture.md) |

---

## Recommendations for Future Phases

1. **Profiling Feature**: Research QEMU profiling capabilities before design
2. **DFG Feature**: Research LLVM IR parsing patterns before design
3. **Fusion Validation Feature**: Research RISC-V instruction encoding before design