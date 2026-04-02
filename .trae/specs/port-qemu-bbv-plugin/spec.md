# Port QEMU BBV Plugin Spec

## Why
当前项目的玄铁 QEMU (版本 8.2) 缺少官方最新 QEMU 中的 `bbv` 插件（用于生成 SimPoint 的基本块向量 Basic Block Vectors）。为了支持相关的性能分析和插桩工作，需要将官方的 `bbv.c` 适配并移植到当前项目的 QEMU 中，并通过实际的 Demo 验证其工作正常。

## What Changes
- 基于官方 `bbv.c` 源码适配移植（上游源码：https://raw.githubusercontent.com/qemu/qemu/refs/heads/master/contrib/plugins/bbv.c）。
- 将适配后的 `bbv.c` 放置于 `third_party/qemu/contrib/plugins/bbv.c`。
- 修改 `bbv.c` 以兼容当前 QEMU 8.2 的 Plugin API：
  - 移除对 QEMU 9.x 新增接口 `qemu_plugin_register_vcpu_tb_exec_cond_cb` 的调用。
  - 替换为 `qemu_plugin_register_vcpu_tb_exec_cb`。
  - 在回调函数 `vcpu_interval_exec` 中增加手动的 `interval` 计数判断逻辑。
- 修改 `/workspace/third_party/qemu/contrib/plugins/Makefile`，将 `bbv` 加入编译列表。
- 编译 QEMU (用户态模式 `riscv64-linux-user`) 和相关的 plugins。
- 创建一个用于测试的 C 语言 demo 程序，包含基础的循环运算以生成基本块。
- 使用项目中提供的 `docker-llvm` 工具链将 demo 编译为 RISC-V 架构的二进制文件。
- 使用编译好的 QEMU 和 `bbv` 插件运行 demo，验证是否成功生成 `.bb` 输出文件。

## Impact
- Affected specs: 增强了项目的 QEMU 插桩分析能力。
- Affected code: 
  - `third_party/qemu/contrib/plugins/bbv.c`
  - `third_party/qemu/contrib/plugins/Makefile`

## ADDED Requirements
### Requirement: 适配并运行 BBV 插件
系统应当能够利用移植后的 `bbv` 插件，在执行目标架构二进制文件时，采集并输出基本块向量。

#### Scenario: Success case
- **WHEN** 用户使用挂载了 `libbbv.so` 的 QEMU 运行 RISC-V 测试程序。
- **THEN** 插件正常执行，并在指定路径生成记录基本块频次的 `.bb` 文件。
