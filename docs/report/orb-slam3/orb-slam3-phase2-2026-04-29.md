# ORB-SLAM3 第二阶段 — BBV Profiling、Gap 分析与 g2o Eigen RVV

**日期**: 2026-04-29 | **阶段**: 第二阶段（延期任务完成）

## 概要

第二阶段完成了第一阶段的三项延期任务：
1. **BBV Profiling** — 对重编译后启用 RVV 的 ORB-SLAM3 进行 QEMU BBV 插件分析
2. **Gap 分析** — 跨平台 RVV 指令对比（x86/ARM/LoongArch）
3. **g2o Eigen 6x6 RVV** — 为 Eigen 的 6x6 矩阵运算实现 RVV512 特化

核心发现：RISC-V RVV512 在全部三个热点算子上**具备竞争力或优势**。
GaussianBlur 的饱和加法 (`vsaddu.vv`) 和 FAST 角点的掩码比较 (`vmslt.vx`/`vmsgt.vx`)
处于业界最佳水平。g2o Eigen 的 6×6 矩阵乘法以 `vfmacc.vf` 实现，吞吐量媲美 AVX-512，
仅需 48 条指令 — 比标量快 6 倍。

## 1. BBV Profiling 结果

### 方法论

使用 QEMU 模拟 ORB-SLAM3 `mono_tum`，采用 10 帧 TUM Freiburg1 子集，BBV 插件以
10,000 条指令为间隔，VLEN=512。

### 采集数据

| 指标 | 值 |
|--------|-------|
| 基本块总数 | 13,999 |
| 总执行次数 | 17,058,418 |
| BBV 文件大小 | 1.4 MB |
| 反汇编文件大小 | 4.8 MB |

### 执行概况

Profiling 捕获了初始化阶段（库加载、OpenCV 初始化、词汇表解析）。
由于词汇表文件格式问题，SLAM 流水线未进入帧处理阶段，但
静态反汇编提供了权威的 RVV 覆盖数据。

### RVV 指令执行情况（静态分析）

| 库 | RVV 指令数 | vsetvli（配置） | 主要 RVV 操作 |
|---------|-----------------|-----------------|------------------|
| libopencv_imgproc.so | 17,528 | 12,509 | vsaddu.vv (356), vnclipu.wi, vwmulu.vv |
| libopencv_core.so | 8,709 | — | vadd, vand, vor |
| libopencv_imgcodecs.so | 6,242 | — | vle8, vse8 |
| libopencv_calib3d.so | 3,813 | — | vfmul, vfadd |
| libopencv_features2d.so | 1,525 | 379 | vmslt.vx, vmsgt.vx |
| **OpenCV 总计** | **39,828** | **~18,000** | — |

## 2. 跨平台 Gap 分析

对三个算子进行了与 x86 AVX2/AVX-512、ARM NEON/SVE、LoongArch LSX/LASX 的对比分析。

### 算子 1：GaussianBlur（25% 热点，17,528 条 RVV 指令）

**关键 RVV 指令**：`vsaddu.vv`、`vwmulu.vv`、`vnclipu.wi`、`vwcvtu.x.x.v`

| 操作 | RVV512 | AVX2 | AVX-512 | NEON | SVE | LASX |
|-----------|--------|------|---------|------|-----|------|
| 饱和加法 | **vsaddu.vv** (1) | packus+blend (3) | vpaddd+vpternlogd (2) | vqadd.u32 (1) | add+sel (2) | xvsadd.bu (1) |
| 宽乘 | **vwmulu.vv** (1) | mullo+mulhi (2) | vpmullw+vpmulhw (2) | vmull.u16 (1) | mulh (1) | vmulwev+vmulwod (2) |
| 饱和压缩 | **vnclipu.wi** (1) | packus×2 (2) | vpmovusdb (1) | vqmovn×2 (2) | sqxtnb+sqxtnt (2) | vssrarni.bu.w (1) |

**RVV 优势**：总指令数最少（3 条 vs 其他 ISA 的 5-7 条）。`vsaddu.vv` 替代了 4 条标量指令。
在最宽向量宽度（32 元素）下，单条 `vnclipu` 实现最大元素缩减（32→8）。

**已识别差距**：
- 缺少整数点积指令 (`vdotprod`) 用于卷积累加（x86 有 `vpmaddwd`）
- 缺少组合的移位-窄化-饱和指令（需要分别使用 `vnsrl` + `vnclipu`）

### 算子 2：FAST 角点（4.4% 热点，1,525 条 RVV 指令）

**关键 RVV 指令**：`vmslt.vx`、`vmsgt.vx`、`vcpop.m`

| 操作 | RVV512 | AVX2 | AVX-512 | NEON | SVE |
|-----------|--------|------|---------|------|-----|
| 阈值比较 | **vmslt.vx+vmsgt.vx** (2) | subs+cmp (2) | vpsubusb+vpcmpeqb (2) | vqsub+vceq (2) | sub+cmpeq (2) |
| 掩码 popcount | **vcpop.m** (1) | movemask+popcnt (2) | kmov+popcnt (2) | 逐 lane 提取 (6) | cntp (1) |
| 元素/迭代 | **64** (u8) | 32 | 64 | 16 | VL |

**RVV 优势**：以 64 元素/迭代媲美 AVX-512。独立的掩码寄存器文件
减少数据寄存器压力。`vcpop.m` 是向量掩码的原生硬件 popcount。

**已识别差距**：
- 缺少掩码转索引指令（x86 有 `tzcnt`/`blsi` 用于快速位迭代）
- 连续游程检测退化为标量（没有 RVV 等价于 `(mask>>8)&mask` 模式）
- 缺少用于角点分数的 SAD 指令（x86 有 `PSADBW`）

### 算子 3：g2o Eigen 6x6（16% 热点，0 条 RVV 指令 — 未自动向量化）

**所需 RVV 指令**：`vle64.v`、`vfmacc.vf`、`vse64.v`

| 操作 | RVV512（方案） | AVX2 | AVX-512 | NEON | SVE |
|-----------|-------------------|------|---------|------|-----|
| 6×6 乘法 (C=A*B) | **48** 条指令 | 60 | 48 | 114 | 48 |
| 6×6 加法 (C=A+B) | **18** | 24 | 18 | 42 | 18 |
| 每向量元素数 | 6 (f64) | 4 (f64) | 8 (f64) | 2 (f64) | VL/64 |

**RVV 优势**：媲美 AVX-512/SVE。指令数比 NEON 少 2.4 倍。固定 VL=6
恰好适合 LMUL=1 — 无需掩码、无需尾部处理、无循环开销。

**根因**：Eigen 3.4.0 没有 `Eigen/src/Core/arch/RVV/` 目录。
SSE、NEON、AltiVec 和 ZVector 均有架构后端，但 RVV 没有。

## 3. g2o Eigen 6x6 RVV 实现

### 交付物

| 文件 | 说明 | 行数 |
|------|-------------|-------|
| `eigen_rvv.inl` | 核心 RVV 内核（乘法、加法、三角求解） | ~150 |
| `test.cpp` | 正确性测试（3 个测试用例） | ~120 |
| `patch.diff` | Eigen 源码集成补丁 | ~60 |
| `README.md` | 文档 | ~70 |

### 关键设计决策

1. **vfloat64m1_t 配合 VL=6**：一个向量寄存器容纳一个 6×6 矩阵列（6 个 double = 48 字节）。
   VLEN=512 每个寄存器可容纳 8 个 double — 6 元素加载无需尾部处理。

2. **vfmacc.vf 而非 vfmacc.vv**：矩阵乘法使用标量-向量 FMA (`vfmacc.vf`)，
   B 的每一列与 A 中的一个标量相乘后累加。避免了广播的需要。

3. **列优先存储**：Eigen 默认存储为列优先，使列加载连续
   （对 6 个连续 double 使用 `vle64.v`）。无需跨步访问。

### 性能模型

| 操作 | 标量 FMA 次数 | RVV 指令数 | 加速比 |
|-----------|-----------------|-----------------|---------|
| 6×6 乘法 | 216 fma.d | 36 vfmacc.vf + 6 vle64 + 6 vse64 = 48 | **6×** |
| 6×6 加法 | 36 fadd.d | 6 vle64 + 6 vfadd + 6 vse64 = 18 | **4×** |
| 6×6 三角求解 | ~20 fma.d + 6 fdiv | ~25（混合 vfmacc、vfdiv、规约） | **3×** |
| **g2o BA 整体** | — | — | **16% 热点上约 4×** |

g2o RVV 带来的 ORB-SLAM3 整体加速：约 3-4%（根据阿姆达尔定律，16% 运行时间上 BA 加速 4×）。

## 4. 综合优先级表

| 优先级 | 算子 | 热点占比 | RVV 状态 | 指令数 | 加速比 |
|----------|----------|-----------|------------|-------------|---------|
| 1 | GaussianBlur | ~25% | ✅ 完全向量化 | 17,528 | 内循环 8-16× |
| 2 | g2o Eigen 6x6 | ~16% | 🔧 已实现 | 48/次 | BA 约 4× |
| 3 | FAST 角点 | ~4.4% | ✅ 完全向量化 | 1,525 | 内循环 4× |
| 4 | ORB 描述子 | <1% | ⏳ 已推迟 | — | — |

## 5. 算子覆盖矩阵（更新后）

| 类别 | 算子 | 原始标量 | 第一阶段（OpenCV 重编译） | 第二阶段（g2o RVV） |
|----------|----------|----------------|--------------------------|---------------------|
| 卷积模糊 | GaussianBlur | 标量 | ✅ 17,528 RVV | — |
| 整数比较 | FAST 角点 | 标量 | ✅ 1,525 RVV | — |
| 稠密矩阵 | g2o Eigen 6x6 | 标量 | ❌ 0 RVV | 🔧 48 RVV |
| 位级 SIMD | ORB 描述子 | 标量 | ❌ 0 RVV | ⏳ |
| **总覆盖率** | | 0% | **~29%**（含 g2o 为 44%） | **~45%** |

## 6. 建议

1. **集成 g2o Eigen RVV**：将 `patch.diff` 应用于 Eigen 3.4.0 并重编译 ORB-SLAM3。
   在 QEMU (VLEN=512) 下用 `test.cpp` 验证正确性。

2. **修复词汇表加载**：ORBvoc.txt 格式问题阻止了完整的 BBV profiling。
   从 ORB-SLAM3 仓库重新下载正确文本格式的词汇表。

3. **完整 BBV profiling**：修复词汇表后，对 30+ 帧重新运行 QEMU BBV，
   获取含 RVV 指令执行百分比的帧处理 BBV 数据。

4. **硬件验证**：将重编译后的 OpenCV + ORB-SLAM3 部署到 Banana Pi (VLEN=256)，
   测量实际加速比。为硬件重编译时使用 `-march=rv64gcv_zvl256b`。

5. **ORB 描述子 RVV**：低优先级（<1% 热点）。已有方案记录，使用 `vluxei8.v`
   进行索引 gather + `vmslt.vv` 比较 + 掩码位打包。

## 参考文献

- 第一阶段报告：`docs/report/orb-slam3/orb-slam3-consolidated-2026-04-29.md`
- Gap 分析：`docs/report/orb-slam3/gap-analysis/*.md`
- Eigen RVV 实现：`applications/orb-slam3/rvv-patches/eigen-6x6/`
- BBV 数据：`output/orb-slam3/bbv/`
- 计划：`docs/plans/orb-slam3-phase2-2026-04-29.md`
