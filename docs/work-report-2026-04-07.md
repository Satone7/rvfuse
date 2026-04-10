<div style="text-align: center; margin-bottom: 3em;">
<h1 style="font-size: 26pt; margin-bottom: 0.3em; border-bottom: none; text-align: center;">RVFuse 项目工作报告</h1>
<p style="font-size: 14pt; color: #6B5344; margin-bottom: 1.5em; text-indent: 0;">RISC-V 指令融合研究平台 — 项目意图与工作进度</p>
<p style="font-size: 12pt; color: #8B7355; text-indent: 0;">2026 年 4 月 7 日</p>
</div>

---

# 一、项目概述

## 1.1 项目意图

RVFuse 是一个 **RISC-V 指令融合（Instruction Fusion）研究平台**，
其核心目标是：

1. **应用性能剖析** — 通过 QEMU 用户态模拟对目标应用
   （如 ONNX Runtime 推理任务）进行 Basic Block Vector (BBV) 采样，
   识别热点函数与基本块
2. **数据流图生成** — 从热点基本块的反汇编代码中自动生成
   Data Flow Graph (DFG)，揭示指令间的数据依赖关系
3. **融合候选发现** — 统计高频指令组合模式，识别具有数据依赖关系
   且适合硬件融合的指令对
4. **融合验证** — 基于模拟器记录的指令数据，计算融合前后的周期计数差异，
   量化性能收益

**研究背景**：RISC-V 作为开源 ISA，其模块化扩展机制天然适合指令融合优化。
  通过识别实际工作负载中的高频指令序列，可以为自定义扩展（Custom Extension）
  设计提供数据驱动的依据。

## 1.2 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 仿真器 | Xuantie QEMU | RISC-V 用户态模拟 + BBV 插件 |
| 编译器 | Xuantie LLVM 13 | RISC-V 指令集准确定义 |
| 推理引擎 | ONNX Runtime | YOLO 目标检测推理工作负载 |
| 容器化 | Docker | RISC-V 原生构建环境隔离 |
| 分析工具 | Python 3 | BBV 分析、热点报告生成 |
| DFG 引擎 | Python 3 | 数据流图生成与 ISA 描述 |
| 构建编排 | Bash 4.0+ | 全流程自动化脚本 |
| 依赖管理 | Git Submodules | 外部工具链版本锁定 |

## 1.3 仓库结构

```
RVFuse/
├── setup.sh               # 全流程编排器(Steps 0-7)
├── prepare_model.sh       # YOLO模型导出与测试数据准备
├── verify_bbv.sh          # QEMU+BBV插件构建验证
├── docs/                  # 架构文档与设计记录
│   └── plans/             # 各功能的设计与实施计划
├── tools/
│   ├── analyze_bbv.py     # BBV热点分析工具(464行)
│   ├── profile_to_dfg.sh  # 剖析→DFG端到端流水线
│   ├── dfg/               # DFG生成引擎(~3400行)
│   │   ├── __main__.py    # CLI入口
│   │   ├── agent.py       # 基本块执行代理
│   │   ├── dfg.py         # DFG数据模型
│   │   ├── filter.py      # 热点过滤
│   │   ├── instruction.py # 指令建模与寄存器流
│   │   ├── parser.py      # 反汇编解析
│   │   ├── output.py      # 多格式输出(DOT/JSON/PNG)
│   │   ├── gen_isadesc.py # llvm-tblgen ISA描述生成器
│   │   ├── isadesc/       # ISA描述模块(RV64I/F/M)
│   │   └── tests/         # 单元测试(~1300行)
│   ├── rv64gcv-onnxrt/     # Docker RISC-V原生构建
│   └── yolo_runner/       # YOLO推理C++ Runner
└── third_party/           # Git Submodules
    ├── qemu/              # Xuantie QEMU
    └── llvm-project/      # Xuantie LLVM
```

# 二、setup.sh 流水线详解

`setup.sh` 是整个项目的核心编排脚本，实现了从零到 DFG 生成的
**8 步全自动化流水线**（Steps 0-7）。脚本采用
**artifact-based skip detection** 机制：每个步骤检查其产出物是否存在，
已完成的步骤自动跳过，支持 `--force` 重跑指定步骤。

## Step 0: 初始化子模块 (Init Submodules)

- **职责**：克隆 Xuantie QEMU 和 LLVM 仓库到 `third_party/`
- **产出物**：`third_party/qemu/.git`、`third_party/llvm-project/.git`
- **选项**：`--shallow` 使用 `--depth 1` 浅克隆，大幅减少下载量
- **依赖**：Git >= 2.30

## Step 1: 准备模型 (Prepare Model)

- **职责**：导出 YOLO11n ONNX 模型并下载测试图像
- **产出物**：`output/yolo11n.ort`、`output/test.jpg`
- **脚本**：`prepare_model.sh`

## Step 2: 构建 QEMU (Build QEMU)

- **职责**：从源码编译 QEMU 并构建 BBV 采样插件
- **产出物**：`third_party/qemu/build/contrib/plugins/libbbv.so`
- **脚本**：`verify_bbv.sh --force-rebuild`

## Step 3: YOLO 构建 (YOLO Build)

- **职责**：通过 Docker 容器进行 RISC-V 原生交叉编译，
  生成 YOLO 推理可执行文件和 sysroot
- **产出物**：`output/yolo_inference`、`output/sysroot/`
- **脚本**：`tools/rv64gcv-onnxrt/build.sh`

## Step 4: BBV 采样 (BBV Profiling)

- **职责**：使用 QEMU 运行 YOLO 推理，BBV 插件采集基本块执行频率
- **产出物**：`output/yolo.bbv.<pid>.bb`
- **核心命令**：

```bash
qemu-riscv64 -L output/sysroot \
  -plugin libbbv.so,interval=100000,\
outfile=output/yolo.bbv \
  ./output/yolo_inference \
  ./output/yolo11n.ort \
  ./output/test.jpg 10
```

- **选项**：`--bbv-interval <N>` 控制采样间隔

## Step 5: 热点报告 (Hotspot Report)

- **职责**：分析 BBV 数据，反汇编二进制，识别热点基本块并生成 JSON 报告
- **产出物**：`output/hotspot.json`
- **脚本**：`tools/analyze_bbv.py`
- **选项**：`--top <N>`（Top N 块）、覆盖率阈值

## Step 6: DFG 生成 (DFG Generation)

- **职责**：从热点基本块生成数据流图，支持多种输出格式
- **产出物**：`output/dfg/` 目录（DOT/JSON/PNG 文件）
- **脚本**：`tools/profile_to_dfg.sh` → `tools/dfg/` 模块
- **ISA 覆盖**：RV64I（基础整数）、RV64M（乘除法）、RV64F（单精度浮点）
- **选项**：`--top <N>`、`--coverage <N>`

### ISA 描述：LLVM 后端指令集解析

DFG 引擎的指令寄存器解析精度取决于 ISA 描述的完整性。
项目构建了从 LLVM 源码自动提取指令定义的管线：

1. `setup_tblgen.sh` 从 Xuantie LLVM 后端编译 `llvm-tblgen` 工具，
   导出 RISC-V 全量指令定义为 JSON 格式
   （`riscv_instrs.json`，约 98MB）
2. `gen_isadesc.py` 解析该 JSON，按扩展谓词筛选指令，
   自动生成 Python ISA 描述模块（如 `isadesc/rv64f.py`、
   `isadesc/rv64m.py`），每条指令的 `RegisterFlow`
   （rd/rs1/rs2/rs3 位置映射）均来自 LLVM 源定义
3. 基础扩展 `rv64i.py` 由手工编写以覆盖伪指令和特殊格式，
   与自动生成的扩展描述统一注册到 ISA Registry

这一机制保证了指令操作数和寄存器语义与 LLVM 后端保持一致，
避免手工维护带来的遗漏和错误。

### 生成策略：脚本生成 + Agent 检查 + Agent 回退

DFG 生成采用分层策略，兼顾效率与鲁棒性：

```
解析基本块反汇编
      ↓
  脚本构建 DFG (build_dfg)
      ↓                     ↓ 未知指令
   成功输出              Agent 回退生成
      ↓                (dfg-generate SKILL)
  Agent 校验 ←←←←←←←←←←←←
  (dfg-check SKILL)
      ↓
  输出 DOT/JSON/PNG
```

1. **脚本优先**：对每个基本块，首先通过 ISA Registry 中的
   `RegisterFlow` 规则脚本化构建 DFG，速度快、可并行
2. **Agent 回退生成**：当遇到 ISA Registry 未覆盖的指令时，
   调用 `dfg-generate` SKILL，由 Claude Agent 根据指令助记符
   和操作数推断寄存器流，生成完整的 DFG 节点和边
3. **Agent 校验**：对已生成的 DFG，调用 `dfg-check` SKILL
   校验 RAW 依赖边是否完整、寄存器来源是否正确。
   校验结果为 advisory（建议性），不阻断流水线

### 输出 SKILL

该步骤产出了两个可复用的 SKILL：

- **dfg-generate**：接收基本块反汇编文本，输出包含节点（指令）
  和边（数据依赖）的 DFG JSON，处理 ISA Registry 外的指令
- **dfg-check**：接收已生成的 DFG，逐节点校验 RAW 依赖关系，
  报告缺失边、多余边和寄存器错误，返回结构化 JSON 裁决

两者均通过 `agent.py` 中的 `AgentDispatcher` 调度，
支持并行处理多个基本块（`ProcessPoolExecutor`）。

## Step 7: 生成报告 (Generate Report)

- **职责**：汇总所有步骤执行状态，写入 `setup-report.txt`
- **产出物**：`setup-report.txt`
- **特性**：始终执行，记录每个步骤的 PASS/FAIL/SKIPPED 状态

## 流水线命令示例

```bash
# 一键运行完整流水线
./setup.sh

# 自定义参数
./setup.sh --shallow \
  --bbv-interval 50000 \
  --top 30 --coverage 85

# 强制重跑特定步骤
./setup.sh --force 2,3     # 重构建QEMU和Docker
./setup.sh --force-all      # 从头全部重跑
```

# 三、工作进度

## 3.1 时间线与里程碑

项目自 **2026 年 3 月 31 日** 启动，至 **2026 年 4 月 7 日**
共完成 172 次提交、11 个合并 PR。

| 时间段 | 里程碑 | PR |
|--------|--------|-----|
| 03/31 | 项目初始化：仓库结构、架构文档、开发规范 | -- |
| 04/01 | Docker LLVM 交叉编译工具链搭建 | #1 |
| 04/02 | QEMU BBV 插件移植与修复 | #2, #3 |
| 04/03 | ONNX Runtime + YOLO RISC-V 构建流水线 | #5 |
| 04/04 | 自动化 setup 脚本（初版） | #8 |
| 04/05 | DFG 生成引擎（核心管线+并行+PNG输出） | #6, #7 |
| 04/06 | 选择性 DFG 生成 + setup.sh 重构 | #9, #10 |
| 04/07 | F+M ISA 扩展 + llvm-tblgen 代码生成 | #11 |

## 3.2 已完成功能模块

### 模块完成度矩阵

| 模块 | 代码量 | 测试 | 状态 |
|------|--------|------|------|
| setup.sh 编排器 | 810行 Shell | artifact检测 | 已完成 |
| analyze_bbv.py | 464行 Python | pytest完备 | 已完成 |
| DFG 生成引擎 | ~1600行 Python | 1300+行测试 | 已完成 |
| ISA 描述 (I/F/M) | ~245行+生成 | 单元测试覆盖 | 已完成 |
| gen_isadesc.py | 327行 Python | -- | 已完成 |
| Docker 构建脚本 | ~200行 Shell | 端到端测试 | 已完成 |
| YOLO C++ Runner | ~300行 C++17 | 集成验证 | 已完成 |
| QEMU BBV 插件 | 补丁方式 | 端到端测试 | 已完成 |

### DFG 引擎架构

DFG 生成引擎是项目的核心研究工具，采用模块化设计：

```
反汇编文本
    ↓
parser.py → instruction.py → agent.py → dfg.py → output.py
(指令解析)   (寄存器流建模)  (基本块代理) (DFG构建) (DOT/JSON/PNG)
```

关键特性：

- **ISA 可扩展**：通过 `isadesc/` 模块 + `gen_isadesc.py`
  （基于 llvm-tblgen）支持新扩展的快速添加
- **寄存器流追踪**：`instruction.py` 建模了完整的寄存器读写语义，
  支持浮点寄存器
- **热点过滤**：`filter.py` 基于 BBV 数据选择性生成 DFG，
  避免冷路径分析开销
- **多格式输出**：DOT（Graphviz）、JSON（结构化）、PNG（可视化）

# 四、当前阶段总结与下一步

## 4.1 已达成的目标

1. **完整的端到端流水线**：从 `git clone` 到 DFG 生成，
   一条命令 `./setup.sh` 即可完成
2. **可重复构建**：基于 Docker + Git Submodules 的确定性构建环境
3. **DFG 核心能力**：支持 RV64I/F/M 三种 ISA 扩展，
   具备可扩展架构
4. **质量保障**：3,800+ 行 Python 代码配有 1,300+ 行单元测试

## 4.2 待开展工作

### 阶段一：融合方案发现与设计

**1. Agent 驱动的融合候选搜索**

在 DFG 生成引擎中引入融合分析 Agent，遍历热点基本块的
数据流图，自动识别满足融合条件的指令序列模式
（如连续的 load-compute-store、乘加链、浮点运算对等）。
Agent 基于 DFG 中节点的数据依赖边和执行频率，
输出候选融合方案及其适用范围。

**2. 硬件约束同步与方案可行性评估**

与硬件组对齐指令设计上的限制（编码空间、译码复杂度、
流水线资源冲突等），将硬件约束建模为过滤规则。
仅保留在硬件层面可实现的融合方案，淘汰不满足时序、
面积或编码约束的候选。

**3. 融合方案设计 Skill 生成**

将通过可行性评估的融合方案自动结构化为
设计文档（Skill），包含：融合前后指令序列对照、
数据依赖关系图、语义等价证明、预估性能收益。
该 Skill 可直接用于后续的模拟验证阶段。

### 阶段二：方案模拟与收益量化

**4. Agent+脚本驱动的全量 BB 模拟**

对具有可行性的融合方案，通过 Agent 配合脚本的方式，
将方案应用到 YOLO 反汇编的所有基本块（而非仅热点块），
统计每个 BB 中可被融合的指令序列数量和分布。

**5. 基于 BBV 的预期收益计算**

结合 BBV 数据中每个基本块的运行次数，
加权计算融合方案的预期收益：

```
预期收益 = Σ (BB_i 中可融合指令数 × BBV_freq(BB_i))
```

输出融合方案的综合收益排名和逐 BB 明细报告。

### 阶段三：扩展与多样化

**6. ISA 扩展支持**

在现有 RV64I/F/M 基础上扩展 DFG 引擎的 ISA 描述，
支持 D（双精度浮点）、C（压缩指令）、V（向量）扩展，
覆盖更完整的 RISC-V 指令集以发现更多融合机会。

**7. 多应用负载分析**

在 YOLO 之外引入更多代表性应用进行 BBV 剖析与 DFG 分析，
如 SPEC CPU2017 基准测试、SQLite 数据库、
图像处理管线等，验证融合方案的通用性和跨应用收益。

---

*本报告基于 2026 年 4 月 7 日的代码库状态生成。*
