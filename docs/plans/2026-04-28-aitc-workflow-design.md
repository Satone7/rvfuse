# AITC Workflow — 长程多智能体协作工作流设计

**Date**: 2026-04-28 | **Status**: Design | **Author**: Satone7 + Claude Opus 4.7

## 1. 概述

`aitc-workflow` 是将 batch2 实践中验证的长程任务工作流和多智能体协作模式抽象为可复用的 SKILL 体系。它解决 batch2 的核心痛点：**任务执行中取得的经验完全凭借 team leader 的上下文和提示词传承，没有被固化**。

### 1.1 三个层级

| 层级 | 安装位置 | 职责 | 示例 |
|------|---------|------|------|
| 全局 SKILL | `~/.claude/skills/` | 最抽象的工作流、最公共的工具 | `aitc-workflow`、`md2pdf`、`claudeception` |
| 项目 SKILL | `.claude/skills/` → `skills/` | 项目中的公共技能和规范 | `qemu-bbv-usage`、`cross-compile-app`、`perf-profiling` |
| 任务 SKILL | `.claude/skills/aitc-task-<batch>/` | 本次任务执行中涌现的微观操作技能 | `bananapi-riscv`、`perf-profiling-v2` |

**三个层级的核心区别**：

- **Plan 文件**（`docs/plans/`）管的是"这次任务**做什么**"：任务清单、角色分配、执行顺序、验收标准
- **任务 SKILL** 管的是"某个操作**怎么做**"：细粒度、单一职责、随执行涌现
- **项目 SKILL** 是跨任务稳定的"怎么做"知识
- **全局 SKILL** 是跨项目通用的模式和工具

### 1.2 方案选择

选择**方案 C：核心 skill + 任务 skill 生成器（2-tier）**。

- `aitc-workflow` 是唯一的全局核心 skill，融合全局+项目感知
- 项目约定通过 CLAUDE.md + 现有项目 skill 表达，不单独成 skill
- 任务 SKILL 是执行产物，由核心 skill 管理其完整生命周期

## 2. 架构

```
全局层 (~/.claude/skills/)
  aitc-workflow/          ← 核心：工作流引擎 + 任务 skill 生命周期管理
  brainstorming/          ← 现有
  skill-creator/          ← 现有
  claudeception/          ← 现有
  ...

项目层 (.claude/skills/ → skills/)
  qemu-bbv-usage/         ← 现有项目 skill（工具使用）
  rvv-op/                 ← 现有项目 skill（领域方法）
  rvv-gap-analysis/       ← 现有项目 skill（领域方法）
  cross-compile-app/      ← 现有项目 skill（领域方法）
  perf-profiling/         ← 现有项目 skill（领域方法）
  guardian/               ← 现有项目 skill（团队角色）
  ...

  aitc-task-<batch>/      ← NEW：任务 skill 目录（执行中涌现）
    ├── bananapi-riscv.md        ← 全新涌现（SSH + 硬件信息）
    ├── perf-profiling-v2.md     ← 补充项目 skill
    └── guardian-b2.md           ← 项目 skill 的参数化实例

  archived/               ← NEW：归档的任务 skill（任务结束后移入）
```

## 3. 核心 skill：aitc-workflow

### 3.1 三种模式

| 模式 | 触发 | 职责 |
|------|------|------|
| **Plan** | 用户提供任务描述 | 生成 plan 文件 + 初始化 task skill 目录 |
| **Execute** | Plan 确认后 | 按 plan 编排执行，管理 task skill 涌现 |
| **Lifecycle** | 任务结束后 | 归档 review → promote/merge/delete |

### 3.2 Plan 模式

**输入**：自然语言任务描述（应用列表、约束条件、模型偏好）

**流程**：
1. 读取项目 CLAUDE.md，获取项目规范、可用 skill 清单
2. 分析每个应用：架构 → 热点算子预估 → 模型选择建议 → 执行顺序推理
3. 与用户对话对齐（调用 brainstorming 模式）：顺序、优先级、资源约束
4. 生成 `docs/plans/<batch>.md`，内容以描述性规划为主，包含：
   - Team 结构（角色、模型选择、执行顺序）
   - 每个 teammate 的任务描述和 Phase 要求
   - 验收标准 checklist
   - 风险和时间线
5. 在 Team Structure 章节包含强制性的 teammate vs subagent 区分说明 + 最小 Agent() 代码示例
6. 用户确认后，创建 `.claude/skills/aitc-task-<batch>/` 空目录

**输出**：
- `docs/plans/<batch>.md` — 完整 plan 文件
- `.claude/skills/aitc-task-<batch>/` — task skill 目录（初始为空或仅含实例类 SKILL）

### 3.3 Execute 模式

```
1. TeamCreate(team_name)
2. 按需生成实例类 task SKILL（如 guardian-b2.md）
3. Spawn Guardian（使用 task SKILL 中的参数）

FOR EACH app IN plan.apps:
  4. 读取 teammate 配置（model、type、isolation）
  5. 从 plan 描述 + 项目 skill 引用组装 teammate prompt
     - 任务描述（来自 plan）
     - 引用的项目 skill 列表
     - 强制 Discoveries 上报指令
     - 已有 task SKILL 引用
  6. Agent(team_name, name, ...) spawn teammate
  7. 创建对应 Task
  8. WAIT teammate 完成
     a. 定期检查 teammate tmux pane，观察 signals
     b. 记录到 .discovery-hints.md
     c. Teammate 报告到达后交叉比对 + 追问
     d. 固化为 task SKILL
  9. Opus verification subagent
     ├── PASS → shutdown → merge → next
     └── FAIL → fix list → rework loop (max 3)

10. Cross-app synthesis（Phase E）
11. Task SKILL 归档提示
12. TeamDelete + Guardian cron cancel
```

#### 3.3.1 Prompt 组装逻辑

Teammate prompt 由以下部分拼接：

1. **角色声明**：来自 plan 的 teammate 配置（名称、应用、模型）
2. **任务描述**：来自 plan 的该 teammate Phase 0-5 要求
3. **工具引用**：所需项目 skill 的列表（`rvv-op`、`qemu-bbv-usage` 等）
4. **已有 task SKILL 引用**：如 `bananapi-riscv.md`
5. **强制上报指令**（见 4.1 层面 1）
6. **上下文提示**：前序 teammate 已完成的分析（供交叉引用）

#### 3.3.2 WAIT 阶段 Lead 主动发现

Lead 在等待 teammate 时通过 tmux 观察，识别以下信号：

| 信号 | 含义 | 行动 |
|------|------|------|
| 命令失败 + 重试成功 | 可能缺少前置步骤 | 记录，待追问 |
| 手动 workaround 后继续 | 项目 skill 可能有误/过时 | 记录，待追问 |
| 工具输出与预期不符但未报错 | 隐式知识 | 记录，待追问 |
| 执行耗时远超预期 | 未记录的性能约束 | 记录，待追问 |

发现信号后记录到 `.claude/skills/aitc-task-<batch>/.discovery-hints.md`，teammate 报告到达后交叉比对。

### 3.4 任务 SKILL 的生命周期

#### 阶段 1：涌现（Execution 中）

3 种触发条件：

| 类型 | 触发 | 示例 |
|------|------|------|
| **new** | 操作无现有 skill 覆盖 | `ssh-to-riscv` → `bananapi-riscv` |
| **supplement** | 项目 skill 信息过时、脚本有 bug、缺场景 | `perf-profiling-v2`（声明 `supplements: perf-profiling`） |
| **instance** | 项目 skill 需要本次任务的具体参数 | `guardian-b2`（声明 `instance-of: guardian`） |

创建前检查是否真正可复用（不是单次日志）。

#### 阶段 2：迭代（Execution 中）

- SKILL 名称可随内容演化（`ssh-to-riscv` → `bananapi-riscv`）
- 每次使用后根据经验追加内容到 Discoveries 章节

#### 阶段 3：归档（任务结束后）

- 任务 skill 目录移到 `.claude/skills/archived/aitc-task-<batch>/`
- Git 提交

#### 阶段 4：Promote（用户触发）

`aitc-workflow` Lifecycle 模式：

1. 列出所有 task SKILL，按类型分组
2. 对 supplement 类：生成与原始项目 skill 的 diff
3. 对 new 类：建议目标层级（项目 vs 全局）
4. 用户逐项决策：
   - **Merge**：调用 skill-creator 将 supplement 合并到目标 skill
   - **Promote**：升级为独立项目/全局 skill
   - **Archive**：标记 archived，保留为参考
   - **Delete**：废弃

## 4. Discoveries 发现与固化机制

### 4.1 层面 1：Prompt 强制上报

每个 teammate 的 prompt 包含：

```
DISCOVERY REPORTING: After each phase, you MUST include a section
in your completion message:

  ## Discoveries
  - New: <any operation where existing skill was wrong/missing>
  - Supplement: <any correction to project skill instructions>
  - None: <if truly nothing>

If "None" is reported but verification finds an unreported issue,
that counts as a FAIL.
```

### 4.2 层面 2：Opus 验证兜底

Verification subagent 的 checklist 包含一项：比对 teammate execution log 和 reported discoveries，判断是否有遗漏。

### 4.3 层面 3：Lead 主动发现

Lead 在 WAIT 阶段通过 tmux 观察 teammate 状态，识别 discovery signals，记录到 `.discovery-hints.md`，teammate 报告后交叉比对并追问。

## 5. 与现有技能的协作

`aitc-workflow` 调用但不替代以下技能：

| 技能 | 调用时机 |
|------|---------|
| `brainstorming` | Plan 模式中与用户对齐需求 |
| `writing-plans` | 生成 plan 文件 |
| `guardian` | Execute 模式中 spawn Guardian |
| `skill-creator` | Promote 阶段 merge/promote task SKILL |
| `claudeception` | Promote 阶段 review session learnings |
| 项目 skill（`rvv-op`、`qemu-bbv-usage` 等） | 在 teammate prompt 中引用 |

## 6. 实现路径

### 6.1 Phase 0：核心 `aitc-workflow` SKILL.md

- 定义 Plan / Execute / Lifecycle 三种模式的完整指令
- 包含 teammate prompt 组装模板、强制上报指令模板
- 包含 task SKILL 模板（new/supplement/instance 三种 frontmatter）

### 6.2 Phase 1：Plan 模式

- 实现从自然语言任务描述到 plan 文件的生成
- 与 brainstorming 的集成

### 6.3 Phase 2：Execute 模式

- 实现 teammate prompt 组装逻辑
- 实现 WAIT 阶段的 discovery 发现流程
- 实现 task SKILL 的创建和更新

### 6.4 Phase 3：Lifecycle 模式

- 实现归档和 Promote 流程
- 与 skill-creator 和 claudeception 的集成

### 6.5 Phase 4：Dogfooding

- 用 `aitc-workflow` 驱动下一批任务，在实战中迭代
