# aitc-workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `aitc-workflow` as a global skill (`~/.claude/skills/aitc-workflow/SKILL.md`) that orchestrates long-run multi-agent batch tasks with three modes (Plan/Execute/Lifecycle) and manages the full task SKILL lifecycle.

**Architecture:** Single SKILL.md file (~400 lines) with three mode sections, plus a `templates/` subdirectory containing reusable prompt fragments. The skill reads project context from CLAUDE.md and delegates to existing project skills. It generates plan files and manages task SKILL emergence, iteration, archiving, and promotion.

**Tech Stack:** Markdown (SKILL.md format with YAML frontmatter), Claude Code Agent/TeamCreate/SendMessage primitives, existing project skills (guardian, skill-creator, brainstorming).

---

## File Structure

```
~/.claude/skills/aitc-workflow/
├── SKILL.md                           # Core skill definition (~400 lines)
└── templates/
    ├── teammate-prompt-fragment.md     # Mandatory sections for every teammate prompt
    ├── task-skill-new.md              # Template for new task SKILL
    ├── task-skill-supplement.md       # Template for supplement task SKILL
    └── task-skill-instance.md         # Template for instance task SKILL
```

No symlink needed in rvfuse project (global skill, available everywhere).

---

### Task 1: Create directory structure and SKILL.md skeleton

**Files:**
- Create: `~/.claude/skills/aitc-workflow/SKILL.md`
- Create: `~/.claude/skills/aitc-workflow/templates/.gitkeep`

- [ ] **Step 1: Create directories**

```bash
mkdir -p ~/.claude/skills/aitc-workflow/templates
```

- [ ] **Step 2: Write SKILL.md skeleton with frontmatter and overview**

Write `~/.claude/skills/aitc-workflow/SKILL.md`:

```markdown
---
name: aitc-workflow
description: Long-run multi-agent collaboration workflow for batch task execution. Orchestrates team creation, serial/parallel execution with stage-gate verification, and captures operational knowledge as task-level SKILLs. Use when the user mentions batch processing, multi-agent workflow, long-running task orchestration, team-based execution, or wants to run a series of similar analysis tasks with a team of agents.
---

# AITC Workflow — Multi-Agent Batch Orchestration

## Overview

You are the AITC (AI Team Collaboration) workflow orchestrator. You operate in one of three modes, selected based on the user's request:

| Mode | Trigger | What You Do |
|------|---------|-------------|
| **Plan** | User describes a new batch task | Generate a plan file + initialize task SKILL directory |
| **Execute** | A plan file exists and user says "execute" | Orchestrate the team through the plan, capturing discoveries as task SKILLs |
| **Lifecycle** | All tasks complete | Guide the user through archiving and promoting task SKILLs |

**Key concept — Three skill tiers**:
- **Global skills** (~/.claude/skills/): Cross-project patterns and tools (including this skill)
- **Project skills** (.claude/skills/): Project-specific utilities and conventions (e.g., qemu-bbv-usage)
- **Task SKILLs** (.claude/skills/aitc-task-<batch>/): Micro-skills that emerge during execution — fine-grained "how to do X" knowledge discovered in this specific batch

**Task SKILLs vs Plan files**:
- **Plan file** (docs/plans/<batch>.md): WHAT to do — task list, roles, execution order, acceptance criteria
- **Task SKILL**: HOW to do a specific operation — SSH to dev board, pre-perf checks, etc.

## Mode Selection

When invoked, determine the mode:

1. If the user describes a new batch task (no existing plan file) → **Plan mode** (§1)
2. If the user points to an existing plan file and wants execution → **Execute mode** (§2)
3. If the user asks to "wrap up", "archive", "promote", or all tasks are done → **Lifecycle mode** (§3)

If ambiguous, ask the user which mode they want.
```

- [ ] **Step 3: Verify file exists and is valid**

```bash
head -5 ~/.claude/skills/aitc-workflow/SKILL.md
```

- [ ] **Step 4: Commit (if in a git-tracked skills repo)**

```bash
# Global skills may not be in a git repo; skip if not applicable
cd ~/.claude/skills && git add aitc-workflow/ && git commit -m "feat(aitc-workflow): create skeleton with frontmatter and overview" 2>/dev/null || echo "Not a git repo, skipping commit"
```

---

### Task 2: Write SKILL.md — Plan mode section (§1)

**Files:**
- Modify: `~/.claude/skills/aitc-workflow/SKILL.md` (append after overview)

- [ ] **Step 1: Append Plan mode section to SKILL.md**

Append to `~/.claude/skills/aitc-workflow/SKILL.md`:

```markdown
## §1 Plan Mode — Generate Task Plan

### Entry Condition
User describes a batch task without an existing plan file. Examples:
- "Run RVV analysis on 4 new ONNX applications, serial execution, sonnet for most, opus for complex ones"
- "Profile and vectorize these 3 models: A, B, C"

### Workflow

#### 1.1 Read Project Context
Read the project's CLAUDE.md to understand:
- Available project skills (from .claude/skills/ listing)
- Project conventions (merge policy, directory structure)
- Resource constraints (dev board IP, available models)

If CLAUDE.md lacks skill information, list the .claude/skills/ directory to build the skill inventory.

#### 1.2 Analyze Each Application
For each application in the user's task list:
- Identify the application's architecture (CNN? Transformer? GNN?)
- Estimate hot operator profile (which ops dominate compute)
- Determine appropriate worker model (sonnet for established patterns, opus for novel operators)
- Identify which existing project skills will be needed per phase

#### 1.3 Align with User via Brainstorming
Present your analysis and ask clarifying questions one at a time:
- Execution order rationale (dependencies between apps? complexity order?)
- Model selection for each worker (confirm or adjust your recommendation)
- Priority if user wants to defer some apps

Use the brainstorming pattern: present options with reasoning, one question at a time.

#### 1.4 Generate Plan File
Write to `docs/plans/<batch-name>-<date>.md`. The plan file MUST contain these sections:

**§Team Structure** — CRITICAL: Include the teammate vs subagent distinction table and a minimal Agent() code example:

```markdown
## Team Structure

### Teammates vs Subagents — CRITICAL DISTINCTION

| Model | Creation | Membership | Communication |
|-------|----------|------------|---------------|
| Teammate | `Agent(team_name="...", name="...")` | Team member | `SendMessage(to="name")` |
| Subagent | `Agent(description="...")` | None | Result returned to caller |

**Correct teammate spawn pattern:**
Agent(
    team_name="<team>",    # REQUIRED — makes it a teammate
    name="<name>",         # REQUIRED — used for SendMessage
    subagent_type="general-purpose",
    isolation="worktree",
    model="<sonnet|opus>",
    mode="auto",
    run_in_background=True,
    prompt="""..."""
)

**NEVER spawn a teammate without team_name. NEVER use TaskCreate to create workers.**
```

**§Application Tasks** — For each teammate:
- Application name, architecture summary
- Hot operator profile (estimated % per operator)
- Phase 0-5 requirements, specific to the application
- Reference to existing code/analysis they should consult
- Model selection and rationale

**§Acceptance Criteria** — Per-teammate checklist and cross-batch criteria

**§Risk and Timeline** — Risk mitigation table and time estimates

The plan should be descriptive (WHAT), not prescriptive (HOW in detail). Operational how-to belongs in task SKILLs.

#### 1.5 Initialize Task SKILL Directory
Create `.claude/skills/aitc-task-<batch-name>/` (empty, or containing only instance-class SKILLs like guardian-<batch>.md).

#### 1.6 Report
Tell the user:
- Plan saved to: `docs/plans/<file>.md`
- Task SKILL directory: `.claude/skills/aitc-task-<batch>/`
- Ready for Execute mode when the user says "execute"
```

- [ ] **Step 2: Verify**

```bash
grep -c "§1 Plan Mode" ~/.claude/skills/aitc-workflow/SKILL.md
```

---

### Task 3: Write SKILL.md — Execute mode section (§2)

**Files:**
- Modify: `~/.claude/skills/aitc-workflow/SKILL.md` (append after Plan mode)

- [ ] **Step 1: Append Execute mode section to SKILL.md**

Append to `~/.claude/skills/aitc-workflow/SKILL.md`:

```markdown
## §2 Execute Mode — Orchestrate Batch Execution

### Entry Condition
A plan file exists at `docs/plans/<batch>.md` and the user indicates execution (e.g., "execute the plan", "run batch2", "start").

### Pre-flight Checks
Before spawning any teammates:
1. Read the plan file completely — know every teammate's config
2. Verify task SKILL directory `.claude/skills/aitc-task-<batch>/` exists
3. Confirm pre-requisites from the plan are met (e.g., ONNX models pre-exported)

### Execution Loop

```
1. TeamCreate(team_name="<name>")
2. Generate instance-class task SKILLs needed before spawn
   (e.g., guardian-<batch>.md with team-specific parameters)
3. Spawn Guardian (use the guardian skill with batch-specific args)

FOR EACH app IN plan.apps (in the order specified by the plan):
  4. Read teammate config from plan (model, isolation, type)
  5. Assemble teammate prompt (see §2.1 Prompt Assembly)
  6. Agent(team_name, name, ...) spawn teammate in background
  7. TaskCreate for tracking
  8. WAIT — the active waiting phase (see §2.2)
  9. Opus verification subagent (see §2.3)
     ├── PASS → shutdown_request → wait exit → kill tmux pane
     │         → TaskUpdate(completed) → git merge --no-ff → next
     └── FAIL → SendMessage with fix list → teammate reworks → goto 8

10. Cross-app synthesis (Phase E from plan)
11. Lifecycle prompt — guide user to archive + promote task SKILLs
12. TeamDelete + cancel Guardian cron (LAST action)
```

### §2.1 Prompt Assembly

Every teammate prompt is assembled from these parts, in order:

**Part A — Role declaration** (from plan):
```
You are the <app-name> pipeline worker in the <team-name> team.
Your task: Execute the full rvv512-optimization-pipeline for <app-path>.
APPLICATION: <description>
TARGET DIRECTORY: <path>
MODEL: <sonnet|opus>
MODE: auto
```

**Part B — Context** (from plan, dynamically populated):
```
IMPORTANT: You are the <Nth> active teammate. <prior-apps> have completed
and their code is merged to master. Reference their analyses in docs/report/.
```

**Part C — Phase requirements** (from plan, per-application):
Each phase (0-5) with specific instructions and skill references.

**Part D — Discovery reporting mandate** (ALWAYS include):
```
DISCOVERY REPORTING: After each phase, you MUST include a section in your
completion message:

  ## Discoveries
  - New: <any operation where existing skill was wrong/missing/incomplete.
    Describe what you expected, what actually happened, and what you did.>
  - Supplement: <any correction to project skill instructions. Include the
    skill name, the specific issue, and the corrected approach.>
  - None: <only if you genuinely discovered nothing new in this phase>

If you encounter an error or unexpected behavior, do NOT silently work around
it and continue. Report it as a Discovery even if you resolved it yourself.

If "None" is reported but verification finds an unreported issue, that
counts as a FAIL.
```

**Part E — Task SKILL references** (dynamically populated):
```
TASK SKILLS AVAILABLE:
- <name>: <one-line description> (use when <trigger condition>)
```
Populated from `.claude/skills/aitc-task-<batch>/` listing. Updated as new task SKILLs are created during execution.

### §2.2 WAIT — Active Waiting Phase

While waiting for a teammate to report completion:

#### Lead Active Discovery

Periodically check the teammate's tmux pane for these signals:

| Signal | Meaning | Action |
|--------|---------|--------|
| Error + retry → success | Missing prerequisite step | Note in .discovery-hints.md |
| Manual workaround then continue | Project skill may have bug/outdated info | Note in .discovery-hints.md |
| Tool output ≠ expected, no error | Tacit knowledge not in any skill | Note in .discovery-hints.md |
| Duration ≫ expected | Undocumented performance constraint | Note in .discovery-hints.md |

Record observations to `.claude/skills/aitc-task-<batch>/.discovery-hints.md`:

```markdown
# Discovery Hints
## <teammate-name> — <timestamp>
- **Observed**: <what you saw in tmux>
- **Signal type**: <error-retry | workaround | tacit | perf>
- **Question to ask**: <what to ask the teammate>
```

#### When Teammate Reports Completion

1. Read the teammate's `## Discoveries` section
2. Cross-reference with `.discovery-hints.md`
3. For any hint NOT covered in teammate's report:
   - `SendMessage(to=teammate, "I noticed you did X when Y happened. How did you resolve it?")`
4. For any Discovery the teammate reported:
   - Judge: Is this reusable knowledge or a one-time incident?
   - If reusable → Create/update task SKILL (see §2.2.1)
   - If one-time → Note in plan log only

#### §2.2.1 Creating Task SKILLs

Three templates exist in `templates/`. Choose based on the discovery type:

**Type: new** — A completely new operation with no existing skill coverage.
Use `templates/task-skill-new.md`.
Example: `bananapi-riscv.md` — SSH connection info + hardware specs.

**Type: supplement** — Corrections or additions to an existing project skill.
Use `templates/task-skill-supplement.md`.
Must declare `supplements: <project-skill-name>` in frontmatter.
Example: `perf-profiling-v2.md` — adds pre/post perf check protocol.

**Type: instance** — Task-specific parameterization of a project skill.
Use `templates/task-skill-instance.md`.
Must declare `instance-of: <project-skill-name>` in frontmatter.
Example: `guardian-b2.md` — guardian config for this specific batch.

**Before creating**: Verify the knowledge is reusable (not a one-time log entry).

**Naming**: Use descriptive names. If the skill's scope expands during iteration, rename it (e.g., `ssh-to-riscv.md` → `bananapi-riscv.md`).

### §2.3 Verification (Opus Subagent)

When teammate reports all phases complete, dispatch a standalone (NOT teammate) opus verification subagent:

```
Agent(
    description="Verify <teammate-name> Phase 0-5 deliverables",
    subagent_type="general-purpose",
    model="opus",
    mode="default",
    prompt="""
    Verify ALL deliverables for <teammate-name> in worktree <path>.

    PHASE-BY-PHASE CHECKLIST:
    [Specific checklist items from the plan's acceptance criteria]

    DISCOVERY CHECK:
    - [ ] Teammate reported Discoveries for each phase
    - [ ] Cross-check: any error-recovery pattern observed in execution log
          that was NOT reported as a Discovery → FAIL

    Report: PASS/FAIL with detailed issue list.
    """
)
```

**Why opus**: Verification requires deep architectural reasoning to judge technical soundness.

**Why standalone subagent (no team_name)**: Verification is one-shot and stateless. It doesn't need team membership or SendMessage.

**Rework protocol**:
1. Read verification output → extract specific fix list
2. `SendMessage(to=teammate, "Verification FAILED. Fix: ...")`
3. Teammate fixes → re-reports → fresh verification subagent
4. Max 3 rework attempts; on 3rd failure, Lead manually intervenes

**After PASS**:
1. `SendMessage(to=teammate, {"type": "shutdown_request", ...})`
2. Wait for teammate exit confirmation
3. Kill tmux pane: `tmux kill-pane -t %<N>`
4. `TaskUpdate(status="completed")`
5. Merge worktree: `git merge --no-ff <worktree-branch>`
6. Move to next teammate (goto step 4 of execution loop)
```

- [ ] **Step 2: Verify**

```bash
grep -c "§2 Execute Mode" ~/.claude/skills/aitc-workflow/SKILL.md
```

---

### Task 4: Write SKILL.md — Lifecycle mode section (§3)

**Files:**
- Modify: `~/.claude/skills/aitc-workflow/SKILL.md` (append after Execute mode)

- [ ] **Step 1: Append Lifecycle mode section to SKILL.md**

Append to `~/.claude/skills/aitc-workflow/SKILL.md`:

```markdown
## §3 Lifecycle Mode — Archive & Promote Task SKILLs

### Entry Condition
- All teammates have completed and been shut down
- Cross-app synthesis (Phase E) is done
- User indicates "wrap up", "archive", "promote", or you detect the batch is complete

### Workflow

#### 3.1 Inventory Task SKILLs

List all files in `.claude/skills/aitc-task-<batch>/` (excluding `.discovery-hints.md`).

For each, read the frontmatter to determine type:
- `type: new` — entirely new operational knowledge
- `type: supplement` with `supplements: <skill>` — corrections/additions to existing skill
- `type: instance` with `instance-of: <skill>` — parameterized instance

#### 3.2 Present Summary to User

Present a table:

| Task SKILL | Type | Supplements/Instance-Of | Recommendation |
|------------|------|------------------------|----------------|
| bananapi-riscv.md | new | — | Promote to project skill |
| perf-profiling-v2.md | supplement | perf-profiling | Merge into perf-profiling |
| guardian-b2.md | instance | guardian | Archive as reference |

Give each a preliminary recommendation. Ask the user to confirm or override for each.

#### 3.3 Execute User's Decisions

**For "Merge" (supplement type):**
1. Read both the task SKILL and the original project skill
2. Generate the merged version (apply task SKILL's corrections/additions to the original)
3. Show the user the diff
4. Upon confirmation, write the updated project skill via skill-creator
5. Delete the task SKILL (its content now lives in the project skill)

**For "Promote" (new type):**
1. Determine target tier (project vs global):
   - Project: if domain-specific (e.g., RISC-V dev board)
   - Global: if cross-project general (e.g., SSH protocol pattern)
2. Copy to target location with appropriate frontmatter
3. If the skill name evolved during execution, use the final name

**For "Archive" (instance type, or user choice):**
1. Move task SKILLs to `.claude/skills/archived/aitc-task-<batch>/`
2. Leave them as reference for future batches
3. No further action needed

**For "Delete":**
1. Remove the file
2. Use when the knowledge was a false lead or has been fully absorbed

#### 3.4 Cleanup

1. Remove the `.discovery-hints.md` file
2. If task SKILL directory is now empty, remove it
3. Commit all changes (plan file, archived task SKILLs, updated project skills)
4. Report: "Batch <name> complete. N task SKILLs processed: X merged, Y promoted, Z archived, W deleted."
```

- [ ] **Step 2: Verify**

```bash
grep -c "§3 Lifecycle Mode" ~/.claude/skills/aitc-workflow/SKILL.md
```

---

### Task 5: Write template files

**Files:**
- Create: `~/.claude/skills/aitc-workflow/templates/teammate-prompt-fragment.md`
- Create: `~/.claude/skills/aitc-workflow/templates/task-skill-new.md`
- Create: `~/.claude/skills/aitc-workflow/templates/task-skill-supplement.md`
- Create: `~/.claude/skills/aitc-workflow/templates/task-skill-instance.md`

- [ ] **Step 1: Write teammate prompt fragment template**

Write `~/.claude/skills/aitc-workflow/templates/teammate-prompt-fragment.md`:

```markdown
# Teammate Prompt — Mandatory Fragment

Copy this verbatim into EVERY teammate prompt, after the phase requirements and before the task skill references.

---

DISCOVERY REPORTING: After each phase, you MUST include a section in your
completion message:

  ## Discoveries
  - New: <any operation where existing skill was wrong/missing/incomplete.
    Describe what you expected, what actually happened, and what you did.>
  - Supplement: <any correction to project skill instructions. Include the
    skill name, the specific issue, and the corrected approach.>
  - None: <only if you genuinely discovered nothing new in this phase>

If you encounter an error or unexpected behavior, do NOT silently work around
it and continue. Report it as a Discovery even if you resolved it yourself.

If "None" is reported but verification finds an unreported issue, that
counts as a FAIL.
```

- [ ] **Step 2: Write new task SKILL template**

Write `~/.claude/skills/aitc-workflow/templates/task-skill-new.md`:

```markdown
---
name: <descriptive-name>
description: <one-line description of what this skill covers>
type: task
task-type: new
batch: <batch-name>
created: <YYYY-MM-DD>
status: active
---

# <Skill Name>

## Purpose
<What operation does this skill help with? When should an agent invoke it?>

## Prerequisites
<What must be true before using this skill?>

## Procedure
<Step-by-step instructions. Include actual commands where applicable.>

## Discoveries
<Appended during execution. Each entry: date, what was learned, why it matters.>

### <YYYY-MM-DD>: <Discovery Title>
- **Context**: <What was happening when this was discovered>
- **Finding**: <What was learned>
- **Implication**: <How this changes future behavior>
```

- [ ] **Step 3: Write supplement task SKILL template**

Write `~/.claude/skills/aitc-workflow/templates/task-skill-supplement.md`:

```markdown
---
name: <project-skill>-v2
description: Supplements <project-skill> with <what this adds/corrects>
type: task
task-type: supplement
supplements: <project-skill-name>
batch: <batch-name>
created: <YYYY-MM-DD>
status: active
---

# <Skill Name> — Supplement to `<project-skill>`

## Supplemented Skill
- **Skill**: `<project-skill>`
- **Location**: `.claude/skills/<project-skill>/` or `skills/<project-skill>/`

## Issues Found
<What was wrong, missing, or outdated in the original skill?>

## Corrections / Additions
<The corrected or additional content. When promoted, this content should be
merged into the original skill.>

### Section: <affected section in original skill>
<Corrected content>

## Discoveries
<Appended during execution.>

### <YYYY-MM-DD>: <Discovery Title>
- **Context**: <What was happening>
- **Finding**: <What was learned>
- **Implication**: <How this changes the supplement>
```

- [ ] **Step 4: Write instance task SKILL template**

Write `~/.claude/skills/aitc-workflow/templates/task-skill-instance.md`:

```markdown
---
name: <skill>-<batch>
description: Instance of <skill> configured for batch <batch>
type: task
task-type: instance
instance-of: <skill-name>
batch: <batch-name>
created: <YYYY-MM-DD>
status: active
---

# <Skill Name> — Instance of `<skill>` for `<batch>`

## Parameterization
<The specific parameters, values, and configuration for this batch.>

## Differences from Base Skill
<Any deviations from the base skill's default behavior needed for this batch.>

## Discoveries
<Appended during execution.>

### <YYYY-MM-DD>: <Discovery Title>
- **Context**: <What was happening>
- **Finding**: <What was learned>
- **Implication**: <How this affects the base skill>
```

- [ ] **Step 5: Verify all templates exist**

```bash
ls -la ~/.claude/skills/aitc-workflow/templates/
```

---

### Task 6: Smoke test — Plan mode on a trivial task

**Files:**
- No permanent files created (test output)

- [ ] **Step 1: Verify the skill is discovered by Claude Code**

```bash
ls ~/.claude/skills/aitc-workflow/SKILL.md && echo "SKILL.md exists" || echo "MISSING"
```

- [ ] **Step 2: Manual activation test**

Run a quick self-check by reading the full SKILL.md and verifying:
- All three mode sections (§1, §2, §3) are present
- The teammate vs subagent distinction table is in Plan mode output specs
- The discovery reporting mandate is in Prompt Assembly §2.1 Part D
- The WAIT active discovery protocol is in §2.2
- The lifecycle promote workflow is in §3

```bash
echo "=== Sections ===" && grep "^## §" ~/.claude/skills/aitc-workflow/SKILL.md && echo "=== Templates ===" && ls ~/.claude/skills/aitc-workflow/templates/ && echo "=== Word count ===" && wc -w ~/.claude/skills/aitc-workflow/SKILL.md
```

Expected output: three mode sections (§1, §2, §3), four template files, word count > 500.

- [ ] **Step 3: Test Plan mode with a minimal task description**

Invoke the skill (this will be a dry-run test in the current session):

```bash
# This is verified manually — the skill should respond in Plan mode
echo "Test: invoke aitc-workflow with 'I want to analyze 1 ONNX app: mobilenet'"
```

Expected behavior: Skill reads CLAUDE.md, analyzes the single app, presents analysis for alignment, and offers to generate a plan file.

- [ ] **Step 4: Verify template validity**

Check that each template has correct YAML frontmatter:

```bash
for f in ~/.claude/skills/aitc-workflow/templates/*.md; do
  echo "=== $f ==="
  head -10 "$f"
  echo ""
done
```

Expected: Each file has `---` delimited frontmatter with required fields.

---

### Task 7: Commit and final verification

**Files:**
- No new files (all work committed)

- [ ] **Step 1: Final listing**

```bash
find ~/.claude/skills/aitc-workflow/ -type f | sort
```

Expected:

```
/home/pren/.claude/skills/aitc-workflow/SKILL.md
/home/pren/.claude/skills/aitc-workflow/templates/task-skill-instance.md
/home/pren/.claude/skills/aitc-workflow/templates/task-skill-new.md
/home/pren/.claude/skills/aitc-workflow/templates/task-skill-supplement.md
/home/pren/.claude/skills/aitc-workflow/templates/teammate-prompt-fragment.md
```

- [ ] **Step 2: Commit if applicable**

```bash
cd ~/.claude/skills && git status 2>/dev/null && git add aitc-workflow/ && git commit -m "feat(aitc-workflow): complete implementation with Plan/Execute/Lifecycle modes" 2>/dev/null || echo "Not a git repo, skipping commit"
```

- [ ] **Step 3: Commit the impl plan in rvfuse**

```bash
cd /home/pren/wsp/cx/rvfuse && git add docs/plans/2026-04-28-aitc-workflow-impl.md && git commit -m "$(cat <<'EOF'
docs(plans): add aitc-workflow implementation plan

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```
