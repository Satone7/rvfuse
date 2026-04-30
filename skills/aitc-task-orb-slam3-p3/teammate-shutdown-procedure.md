# Teammate Shutdown — Complete Procedure

## Problem

`SendMessage` with `shutdown_request` sends a protocol-level shutdown message, and the teammate may approve it, but **the tmux pane remains alive**. The agent process stays at an idle prompt, consuming resources and cluttering the tmux session.

## Root Cause

The `SendMessage` protocol operates at the Claude Code application layer — it tells the teammate's agent to exit its task loop. But the underlying tmux pane (which runs the `claude` CLI process) is NOT automatically terminated. The pane will sit idle at `❯` prompt indefinitely.

## Correct Shutdown Procedure

For EACH teammate that completes its task, the Lead must execute ALL THREE steps:

### Step 1: Send shutdown request
```
SendMessage(to="<teammate-name>", message={"type":"shutdown_request","reason":"..."})
```

### Step 2: Wait for approval
The teammate responds with `{"type":"shutdown_response","approve":true}`.

### Step 3: Kill the tmux pane
```bash
# Find pane ID from team config
cat ~/.claude/teams/<team-name>/config.json | python3 -c "import sys,json; [print(m['name'],m['tmuxPaneId']) for m in json.load(sys.stdin)['members']]"

# Kill the pane
tmux kill-pane -t <pane_id>
```

## Verification

After killing, verify the pane is gone:
```bash
tmux list-panes -a -F '#{pane_id} #{pane_current_command}'
```

The teammate's pane_id should NOT appear in the list.

## Worktree Cleanup

After killing the pane, also clean up the worktree:
```bash
git -C /home/pren/wsp/cx/rvfuse worktree remove /tmp/worktrees/<team>-<task> --force
```

## Team Config Cleanup

After killing the pane, the team config still shows the member. The `isActive` field changes to `false` after shutdown approval, but the member entry persists. This is acceptable — the config serves as an execution log.

## Self-Maintenance Rule

If you discover that any instruction in this file is wrong, incomplete, or produces incorrect results, edit this file directly: correct the affected section, append to `## Discoveries` with what you found and changed.
