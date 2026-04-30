---
name: guardian-orb-slam3-p2
description: Guardian instance for orb-slam3 Phase 2 — monitors orb-slam3-p2 team via cron loop
type: instance
instance-of: guardian
---

# Guardian Instance — orb-slam3 Phase 2

## Parameterization

- team_name: orb-slam3-p2
- batch_name: orb-slam3-phase2
- instance_skill_path: skills/aitc-task-orb-slam3/guardian-orb-slam3-p2.md
- log_file_path: docs/plans/guardian-log-orb-slam3-p2.md
- notes_file_path: /tmp/guardian-orb-slam3-p2-notes.txt
- plan_file_path: docs/plans/orb-slam3-phase2-2026-04-29.md
- task_count: 4
- cron_interval: "*/5 * * * *"

## Differences from Base Skill

None — follows base skill exactly.

## Discoveries

### 2026-04-29 — Protocol Gap: Orphaned Guardian
**Observed:** Work session ended with TeamDelete but Guardian cron was NOT cancelled first. Lead pane disappeared, team/task directories deleted, but Guardian cron continued firing with no Lead to monitor.

**Protocol implication:** Base skill states "Cancellation is Lead's last action" but doesn't handle abnormal termination where Lead session ends without cleanup.

**Recommended amendment:** Add Rule 0 (pre-check): "If Lead pane does not exist AND team directory deleted → Guardian logs abnormal termination and notes that manual CronDelete is required. Continue firing until session ends or manual cancellation."

### 2026-04-30 — Critical Logic Fix: Deadlock Prevention & Pane-Only Verification
**Observed:** Guardian's "teammates running" UI indicator could refer to Guardian itself (running cron tick), creating logical deadlock: Lead waits on "teammates" (Guardian), Guardian sees "teammates running" and assumes workers exist, doesn't intervene.

**Root cause:** Guardian was relying on Lead's UI indicator instead of actual pane verification.

**User correction:** Guardian must verify worker state ONLY via tmux panes:
1. Check if Lead is idle via pane capture
2. Check if any **worker pane** (excluding Guardian) exists AND is actually working
3. **Rule 7 applies ONLY if condition 2 is true** (worker pane exists and working)

**Protocol amendment:**
- **NEVER use `ps` processes** to verify workers — may have residual/irrelevant processes
- **Only use tmux panes** for teammate verification
- **Exclude Guardian (self)** from worker count — Guardian's cron tick makes Lead show "teammates running"
- **Edge case:** If no worker pane but Lead shows background task (e.g., "@et1" in status) with actual progress → Lead is working via background agents, NOT deadlock. Rule 7 applies (everything normal).

### 2026-04-30 — Inbox Collapse & Background Task Detection
**Observed:** Guardian sent SendMessage (Tick #92) about "No worker panes". Lead received it (status shows "@guardian @guardian") but inbox was collapsed ("shift+↓ to expand"). Lead continued working via @et1 background task and produced actual output (gap analysis table, ~45% coverage).

**Lesson:** 
- Guardian messages queue in Lead's inbox but may be collapsed
- Lead having "@et1" or similar in status line indicates background agents working
- Lead displaying actual work output (tables, progress numbers) = NOT deadlock
- **SendMessage intervention may not be needed** if Lead has visible progress from background tasks

**Refinement:** Before Rule 2 intervention, check Lead's pane for:
1. Background task indicators in status line (@et1, @verification, etc.)
2. Actual work output in scrollback (tables, progress, deliverables)
3. If either present → Lead working via background agents → Rule 7, not Rule 2

### 2026-04-30 — ERROR: False Positive on Background Agent Detection
**Observed:** Tick #94 incorrectly assumed "@et1-verification" in status line meant active background agent. Tick #95 verified:
- No et1 process found
- No et1 worktree found
- "@et1-verification" was HISTORICAL tag from completed task, not active agent
- "~45% coverage" output was HISTORICAL, not realtime progress

**Error cause:** Assumed status line tags indicate active agents without verification.

**Correct verification for "background agent running":**
- Check `ps aux | grep <agent-name>` for actual process
- Check `~/.claude/worktrees/` for worktree existence
- If both empty → tag is historical, NOT active → Rule 2 applies

**Protocol correction:**
- Status line tags (@et1, etc.) may be historical residue
- Scrollback output may be completed work, not realtime progress
- Must verify with process/worktree check BEFORE assuming Rule 7
- If no verification evidence → default to Rule 2

### 2026-04-30 — Anti-Spam Rule for Repeated Interventions
**Observed:** Tick #92 and #95 both sent Rule 2 intervention (SendMessage to Lead). Lead did not respond ( inbox collapsed). Tick #96 would have sent same message again, creating spam.

**Protocol addition:**
- If Rule 2 intervention was sent in previous tick and Lead has not responded → DO NOT resend same intervention
- Note state as "Rule 2 condition persists, intervention already sent, awaiting Lead response"
- Only intervene again if:
  1. State has materially changed
  2. New intervention rule applies (e.g., Rule 4 if tasks now complete)
  3. Previous intervention was acknowledged/resolved but new issue appeared

### 2026-04-30 — Escalation: tmux send-keys when SendMessage Ineffective
**Observed:** SendMessage interventions (Tick #92, #95) sent to Lead's inbox but inbox remained collapsed. Lead output unchanged across multiple ticks → SendMessage was not received/acted upon.

**User directive:** If SendMessage ineffective (Lead pane shows no progress), escalate to `tmux send-keys` to inject message directly into Lead's active prompt, bypassing collapsed inbox.

**Protocol escalation ladder:**
1. **First intervention:** SendMessage to Lead (queues to inbox)
2. **If ineffective after 1-2 ticks:** Escalate to `tmux send-keys -t %0 "[Guardian] <message>" Enter`
3. **Message format:** Prefix with `[Guardian]` to identify source

**Implementation:**
```bash
tmux send-keys -t %0 "[Guardian] Lead idle with no workers. Work appears complete (~45% coverage). Spawn workers for remaining tasks or proceed to Lifecycle closure." Enter
```

**Rationale:** tmux send-keys ensures immediate visibility in Lead's active pane, bypassing collapsed inbox. This is the escalation method when SendMessage fails.

### 2026-04-30 — CRITICAL: tmux send-keys requires Enter to submit
**Observed:** Tick #98 used `tmux send-keys -t %0 "[Guardian] message" Enter` but message stayed at prompt, NOT submitted. Lead showed message at prompt line but remained idle.

**Root cause:** First Enter in the command was interpreted as part of the message text, not as submit action. Message sat at prompt waiting for another Enter.

**Correction (Tick #100):**
- Message was visible at prompt: `❯ [Guardian] Lead idle...`
- Sent separate Enter: `tmux send-keys -t %0 Enter`
- Lead immediately started processing: "Spelunking… (11s)"

**Protocol refinement:**
- **Check prompt state BEFORE assuming intervention successful**
- tmux send-keys sends text to prompt, but may need **additional Enter** to submit
- **Verification:** Check if message is still at prompt (shows `❯ [Guardian]...`) vs. Lead actively processing (shows "Spelunking…" or spinner)
- **Two-step process:**
  1. `tmux send-keys -t %0 "[Guardian] message"` — inject text to prompt
  2. `tmux send-keys -t %0 Enter` — submit message to Lead
- **Or use C-m (carriage return):** `tmux send-keys -t %0 "[Guardian] message" C-m`
