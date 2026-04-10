# BBV 插件独立编译方案 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将定制版 BBV 插件（.disas 输出 + exit flush）从 QEMU 子模块中解耦，通过 `tools/bbv/Makefile` 独立编译，同时更新项目脚本适配。

**Architecture:** 在 `tools/bbv/` 下放置定制版 `bbv.c` 和 `Makefile`，基于官方 QEMU 的 `bbv.c` 增加 `.disas` 输出和 `plugin_flush()` 功能。QEMU 子模块保持纯净不修改。官方插件和定制版同时构建。

**Tech Stack:** C (QEMU plugin API), GNU Make, Bash, glib-2.0

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/bbv/bbv.c` | **Create** | Custom BBV plugin with .disas output + plugin_flush() |
| `tools/bbv/Makefile` | **Create** | Independent build for libbbv.so |
| `verify_bbv.sh` | **Modify** | Build custom plugin, update test paths |
| `setup.sh` | **Modify** | Update artifact paths and plugin_so references |
| `patches/qemu-bbv/` | **Delete** | No longer needed |

---

### Task 1: 创建 `tools/bbv/bbv.c` — 定制版 BBV 插件

**Files:**
- Create: `tools/bbv/bbv.c`

Based on `third_party/qemu/contrib/plugins/bbv.c` (official QEMU v9.2.4), with three additions:
1. `.disas` file output (writes assembly for each BB)
2. `plugin_flush()` — flush remaining counts at exit when total < interval
3. Keep 0-based BB indexing (official default)

```c
/*
 * Generate basic block vectors for use with the SimPoint analysis tool.
 * SimPoint: https://cseweb.ucsd.edu/~calder/simpoint/
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Extended: .disas file output and plugin_flush() at exit.
 */

#include <stdio.h>
#include <glib.h>

#include <qemu-plugin.h>

typedef struct Bb {
    uint64_t vaddr;
    struct qemu_plugin_scoreboard *count;
    unsigned int index;
} Bb;

typedef struct Vcpu {
    uint64_t count;
    FILE *file;
} Vcpu;

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;
static GHashTable *bbs;
static GRWLock bbs_lock;
static char *filename;
static FILE *disas_file;
static struct qemu_plugin_scoreboard *vcpus;
static uint64_t interval = 100000000;

static void plugin_flush(void)
{
    for (int i = 0; i < qemu_plugin_num_vcpus(); i++) {
        Vcpu *vcpu = qemu_plugin_scoreboard_find(vcpus, i);
        GHashTableIter iter;
        void *value;

        if (!vcpu->file || !vcpu->count) {
            continue;
        }

        fputc('T', vcpu->file);

        g_rw_lock_reader_lock(&bbs_lock);
        g_hash_table_iter_init(&iter, bbs);

        while (g_hash_table_iter_next(&iter, NULL, &value)) {
            Bb *bb = value;
            uint64_t bb_count = qemu_plugin_u64_get(bb_count_u64(bb), i);

            if (!bb_count) {
                continue;
            }

            fprintf(vcpu->file, ":%u:%" PRIu64 " ", bb->index, bb_count);
            qemu_plugin_u64_set(bb_count_u64(bb), i, 0);
        }

        g_rw_lock_reader_unlock(&bbs_lock);
        fputc('\n', vcpu->file);
    }
}

static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    plugin_flush();

    for (int i = 0; i < qemu_plugin_num_vcpus(); i++) {
        fclose(((Vcpu *)qemu_plugin_scoreboard_find(vcpus, i))->file);
    }

    g_hash_table_unref(bbs);
    g_free(filename);
    if (disas_file) {
        fclose(disas_file);
    }
    qemu_plugin_scoreboard_free(vcpus);
}

static void free_bb(void *data)
{
    qemu_plugin_scoreboard_free(((Bb *)data)->count);
    g_free(data);
}

static qemu_plugin_u64 count_u64(void)
{
    return qemu_plugin_scoreboard_u64_in_struct(vcpus, Vcpu, count);
}

static qemu_plugin_u64 bb_count_u64(Bb *bb)
{
    return qemu_plugin_scoreboard_u64(bb->count);
}

static void vcpu_init(qemu_plugin_id_t id, unsigned int vcpu_index)
{
    g_autofree gchar *vcpu_filename = NULL;
    Vcpu *vcpu = qemu_plugin_scoreboard_find(vcpus, vcpu_index);

    vcpu_filename = g_strdup_printf("%s.%u.bb", filename, vcpu_index);
    vcpu->file = fopen(vcpu_filename, "w");
}

static void vcpu_interval_exec(unsigned int vcpu_index, void *udata)
{
    Vcpu *vcpu = qemu_plugin_scoreboard_find(vcpus, vcpu_index);
    GHashTableIter iter;
    void *value;

    if (!vcpu->file) {
        return;
    }

    vcpu->count -= interval;

    fputc('T', vcpu->file);

    g_rw_lock_reader_lock(&bbs_lock);
    g_hash_table_iter_init(&iter, bbs);

    while (g_hash_table_iter_next(&iter, NULL, &value)) {
        Bb *bb = value;
        uint64_t bb_count = qemu_plugin_u64_get(bb_count_u64(bb), vcpu_index);

        if (!bb_count) {
            continue;
        }

        fprintf(vcpu->file, ":%u:%" PRIu64 " ", bb->index, bb_count);
        qemu_plugin_u64_set(bb_count_u64(bb), vcpu_index, 0);
    }

    g_rw_lock_reader_unlock(&bbs_lock);
    fputc('\n', vcpu->file);
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    uint64_t n_insns = qemu_plugin_tb_n_insns(tb);
    uint64_t vaddr = qemu_plugin_tb_vaddr(tb);
    Bb *bb;

    g_rw_lock_writer_lock(&bbs_lock);
    bb = g_hash_table_lookup(bbs, &vaddr);
    if (!bb) {
        bb = g_new(Bb, 1);
        bb->vaddr = vaddr;
        bb->count = qemu_plugin_scoreboard_new(sizeof(uint64_t));
        bb->index = g_hash_table_size(bbs);
        g_hash_table_replace(bbs, &bb->vaddr, bb);

        if (disas_file) {
            fprintf(disas_file, "BB %u (vaddr: 0x%" PRIx64 ", %" PRIu64 " insns):\n",
                    bb->index, bb->vaddr, n_insns);
            for (size_t i = 0; i < n_insns; i++) {
                struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
                uint64_t insn_vaddr = qemu_plugin_insn_vaddr(insn);
                char *disas = qemu_plugin_insn_disas(insn);
                fprintf(disas_file, "  0x%" PRIx64 ": %s\n", insn_vaddr, disas ? disas : "unknown");
                g_free(disas);
            }
            fprintf(disas_file, "\n");
        }
    }
    g_rw_lock_writer_unlock(&bbs_lock);

    qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
        tb, QEMU_PLUGIN_INLINE_ADD_U64, count_u64(), n_insns);

    qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
        tb, QEMU_PLUGIN_INLINE_ADD_U64, bb_count_u64(bb), n_insns);

    qemu_plugin_register_vcpu_tb_exec_cond_cb(
        tb, vcpu_interval_exec, QEMU_PLUGIN_CB_NO_REGS,
        QEMU_PLUGIN_COND_GE, count_u64(), interval, NULL);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        char *opt = argv[i];
        g_auto(GStrv) tokens = g_strsplit(opt, "=", 2);
        if (g_strcmp0(tokens[0], "interval") == 0) {
            interval = g_ascii_strtoull(tokens[1], NULL, 10);
        } else if (g_strcmp0(tokens[0], "outfile") == 0) {
            filename = tokens[1];
            tokens[1] = NULL;
        } else {
            fprintf(stderr, "option parsing failed: %s\n", opt);
            return -1;
        }
    }

    if (!filename) {
        fputs("outfile unspecified\n", stderr);
        return -1;
    }

    g_autofree gchar *disas_filename = g_strdup_printf("%s.disas", filename);
    disas_file = fopen(disas_filename, "w");

    bbs = g_hash_table_new_full(g_int64_hash, g_int64_equal, NULL, free_bb);
    vcpus = qemu_plugin_scoreboard_new(sizeof(Vcpu));
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    qemu_plugin_register_vcpu_init_cb(id, vcpu_init);
    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);

    return 0;
}
```

- [ ] **Step 1: Create `tools/bbv/bbv.c`**

Copy the code above into `tools/bbv/bbv.c`. Key changes from official:
- Added `static FILE *disas_file;` (line ~30)
- Added `plugin_flush()` function (lines ~33-68) — mirrors `vcpu_interval_exec` logic but calls from `plugin_exit`
- `plugin_exit` calls `plugin_flush()` first, then closes `disas_file`
- `qemu_plugin_install` opens the `.disas` file alongside the `.bb` file
- `vcpu_tb_trans` writes disassembly info when a new BB is first discovered

**Differences from official `third_party/qemu/contrib/plugins/bbv.c`:**
- Lines with `+` comments above: `disas_file` declaration, `plugin_flush`, disas open/write/close
- BB index: `g_hash_table_size(bbs)` — 0-based (same as official)

- [ ] **Step 2: Verify bbv.c compiles**

```bash
cd tools/bbv && make -n
```

Expected: prints the gcc command that would compile bbv.c to libbbv.so with correct includes. Don't run `make` yet — QEMU submodule may not be initialized. Skip if `third_party/qemu/include/qemu/qemu-plugin.h` doesn't exist.

---

### Task 2: 创建 `tools/bbv/Makefile`

**Files:**
- Create: `tools/bbv/Makefile`

```makefile
# Standalone build for custom BBV plugin with .disas output + exit flush.
# Uses qemu-plugin.h from the QEMU submodule (read-only reference).
#
# Usage:
#   make -C tools/bbv/          # build libbbv.so
#   make -C tools/bbv/ clean    # remove built artifacts

CC      ?= gcc
CFLAGS   := -shared -fPIC -O2 -Wall -Werror
CFLAGS  += $(shell pkg-config --cflags glib-2.0)
LDFLAGS  := $(shell pkg-config --libs glib-2.0)

QEMU_HDR = ../../third_party/qemu/include/qemu

libbbv.so: bbv.c $(QEMU_HDR)/qemu-plugin.h
	$(CC) $(CFLAGS) -I$(QEMU_HDR) -o $@ bbv.c $(LDFLAGS)

.PHONY: clean
clean:
	rm -f libbbv.so
```

**Important:** The Makefile uses `pkg-config` for glib-2.0. If the system doesn't have it, the user needs `libglib2.0-dev` (Debian/Ubuntu) or `glib2-devel` (Fedora/RHEL).

- [ ] **Step 1: Create `tools/bbv/Makefile`**

Copy the content above. Use a real tab character before `$(CC)` — not spaces.

- [ ] **Step 2: Test Makefile syntax**

```bash
cd /home/pren/wsp/rvfuse && make -n -C tools/bbv/
```

Expected: prints the gcc command. If it fails with `pkg-config` errors, the user needs to install glib-2.0 dev packages first.

---

### Task 3: 更新 `verify_bbv.sh`

**Files:**
- Modify: `verify_bbv.sh`

Changes:
1. Add `CUSTOM_LIBBBV_SO="${WORKSPACE}/tools/bbv/libbbv.so"` variable (after `PLUGIN_SO` line ~19)
2. After the QEMU `make plugins` step (~line 80), add `make -C "${WORKSPACE}/tools/bbv/"` to build the custom plugin
3. Update the demo run to use `CUSTOM_LIBBBV_SO` instead of `PLUGIN_SO`
4. Update success message to mention both plugins

**Exact changes to `verify_bbv.sh`:**

Add after line 19 (`PLUGIN_SO=...`):
```bash
CUSTOM_LIBBBV_SO="${WORKSPACE}/tools/bbv/libbbv.so"
```

Add after line 80 (`make plugins`), before the `cd "${QEMU_DIR}"`:
```bash
    echo "Building custom BBV plugin (tools/bbv/)..."
    make -C "${WORKSPACE}/tools/bbv/"
```

Change line 19's `PLUGIN_SO` usage in the demo run (line ~112) from `PLUGIN_SO` to `CUSTOM_LIBBBV_SO`:
```bash
# Before:
${QEMU_BIN} -plugin "${PLUGIN_SO}",interval=10000,outfile="${BBV_OUT}" "${DEMO_ELF}"
# After:
${QEMU_BIN} -plugin "${CUSTOM_LIBBBV_SO}",interval=10000,outfile="${BBV_OUT}" "${DEMO_ELF}"
```

Update success check (line ~87) to also check custom plugin:
```bash
# Before:
if [ -f "${QEMU_BIN}" ] && [ -f "${PLUGIN_SO}" ]; then
# After:
if [ -f "${QEMU_BIN}" ] && [ -f "${PLUGIN_SO}" ] && [ -f "${CUSTOM_LIBBBV_SO}" ]; then
```

Update message (line ~88):
```bash
# Before:
    echo "[OK] QEMU and BBV plugin built successfully."
# After:
    echo "[OK] QEMU, official libbbv.so, and custom libbbv.so built successfully."
```

- [ ] **Step 1: Apply the edits to `verify_bbv.sh`**

Make all four changes above. Verify with:
```bash
grep -n 'CUSTOM_LIBBBV_SO\|tools/bbv' verify_bbv.sh
```

---

### Task 4: 更新 `setup.sh`

**Files:**
- Modify: `setup.sh`

Two changes needed:
1. `STEP2_ARTIFACTS` — add custom plugin path
2. `step4_bbv_profiling` — update `plugin_so` path

**Change 1:** After line 49 (`"third_party/qemu/build/contrib/plugins/bbv.so"`), add:
```bash
    "tools/bbv/libbbv.so"
```

**Change 2:** In `step4_bbv_profiling` (~line 549), change:
```bash
# Before:
local plugin_so="${PROJECT_ROOT}/third_party/qemu/build/contrib/plugins/bbv.so"
# After:
local plugin_so="${PROJECT_ROOT}/tools/bbv/libbbv.so"
```

- [ ] **Step 1: Apply the two edits to `setup.sh`**

Make both changes. Verify with:
```bash
grep -n 'libbbv.so\|plugin_so' setup.sh
```

Expected output:
```
49:    "third_party/qemu/build/contrib/plugins/bbv.so"
50:    "tools/bbv/libbbv.so"
549:    local plugin_so="${PROJECT_ROOT}/tools/bbv/libbbv.so"
```

(line numbers may shift slightly)

---

### Task 5: 删除 `patches/qemu-bbv/` 目录

**Files:**
- Delete: `patches/qemu-bbv/0001-feat-plugins-add-basic-block-vector-bbv-plugin-for-Q.patch`
- Delete: `patches/qemu-bbv/0002-feat-plugins-output-basic-block-assembly-instruction.patch`
- Delete: `patches/qemu-bbv/0003-fix-bbv-flush-remaining-BBV-data-at-exit-when-instru.patch`
- Delete: `patches/qemu-bbv/` (directory)

- [ ] **Step 1: Remove the patches directory**

```bash
rm -rf /home/pren/wsp/rvfuse/patches/qemu-bbv/
```

Verify:
```bash
ls /home/pren/wsp/rvfuse/patches/ 2>/dev/null
```

Should be empty or directory doesn't exist (if this was the only subdirectory).

---

### Task 6: 提交验证

**Files:**
- No new files — this is a verification step

- [ ] **Step 1: Verify all changes are consistent**

```bash
# Check no references to old bbv.so path in active scripts
grep -rn 'build/contrib/plugins/bbv\.so' --include='*.sh' .
```

Expected: should only match `setup.sh` line in `STEP2_ARTIFACTS` (which is intentional — the official plugin artifact). Should NOT match any `plugin_so` or `-plugin` usage.

- [ ] **Step 2: Verify the demo can run (if QEMU is already built)**

If `third_party/qemu/build/qemu-riscv64` exists:
```bash
cd /home/pren/wsp/rvfuse
make -C tools/bbv/
./verify_bbv.sh
```

Should show:
- QEMU already built (skip)
- Custom BBV plugin built
- Demo runs successfully
- BBV output generated with `T:` prefix

- [ ] **Step 3: Commit**

```bash
git add tools/bbv/bbv.c tools/bbv/Makefile verify_bbv.sh setup.sh
git rm -r patches/qemu-bbv/
git commit -m "feat(bbv): decouple custom BBV plugin from QEMU submodule

- Add tools/bbv/bbv.c with .disas output and plugin_flush()
- Add tools/bbv/Makefile for independent plugin compilation
- Update verify_bbv.sh and setup.sh to build and use custom libbbv.so
- Remove patches/qemu-bbv/ (no longer needed)
- QEMU submodule remains clean; official plugin also retained"
```

---

## Downstream Impact Summary

| File | Needs change? | Reason |
|------|--------------|--------|
| `tools/analyze_bbv.py` | **No** | Uses `vaddr` for BB matching, `bb_id` is just a label |
| `tools/profile_to_dfg.sh` | **No** | `.disas` path auto-inference works the same |
| `tools/dfg/` (all) | **No** | BB filtering uses `vaddr` from JSON report |
| `.claude/skills/qemu-usage/SKILL.md` | **No** (deferred) | References will be updated when the new pipeline is verified working |
