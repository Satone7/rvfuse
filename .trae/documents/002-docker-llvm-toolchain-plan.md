# 实施计划: Docker LLVM RISC-V 工具链 (002-docker-llvm-toolchain)

## 摘要 (Summary)
本项目旨在为硬件性能受限的开发者提供基于 Docker 的 LLVM RISC-V 交叉编译工具链，以替代直接从源码编译 submodule 中的 LLVM。本计划将严格按照 `specs/002-docker-llvm-toolchain/tasks.md` 中规划的 T001-T041 任务分阶段实施，完成工具链的构建、测试以及文档编写工作。

## 当前状态分析 (Current State Analysis)
- **分支状态**：当前已处于 `002-docker-llvm-toolchain` 分支。
- **文档状态**：相关的设计文档（`spec.md`, `design.md`, `research.md`, `tasks.md` 等）已在 `specs/002-docker-llvm-toolchain/` 目录下就绪。
- **目录结构**：代码库中目前尚未创建 `tools/docker-llvm` 和 `tests/tools` 目录。
- **架构决策**：基于 `research.md`，Docker 镜像将使用 `debian:bullseye-slim` 为基础镜像，并通过 `apt.llvm.org` 安装 LLVM 13 工具链，以保证与项目中 submodule (llvmorg-13.0.0-rc1) 版本的最大兼容性。

## 提议的变更 (Proposed Changes)

我们将分为 6 个阶段逐步推进：

### 阶段 1: 初始化项目 (Phase 1: Setup)
- **T001-T002**: 创建 `tools/docker-llvm/` 和 `tests/tools/` 目录结构。
- **T003**: 在 `tools/docker-llvm/Dockerfile` 编写 Dockerfile。将安装 `clang-13`, `lld-13`, `llvm-13`，并创建 `clang`, `clang++`, `lld`, `llvm-objdump`, `llvm-strip` 等命令的软链接。
- **T004**: 编写 `tools/README.md`，提供工具链的概览信息。

### 阶段 2: 基础架构建设 (Phase 2: Foundational)
- **T005-T009**: 编写 `tools/docker-llvm/common.sh`，实现所有封装脚本依赖的核心逻辑：
  - 检查宿主机 Docker 是否可用。
  - 实现 Docker 镜像的拉取/本地构建函数。
  - 实现统一的错误处理及提示信息。
  - 实现容器内的路径映射（Volume mounting）和用户权限映射（User mapping）。
  - 提供核心的 `docker run` 封装函数供上层脚本调用。

### 阶段 3: 用户故事 1 - 编译 MVP (Phase 3: User Story 1)
- **T010-T012**: 编写测试脚本 `tests/tools/test-docker-llvm.sh`，添加 Docker 可用性测试和基础的 C 代码编译为 RISC-V ELF 的测试。
- **T013-T016**: 编写 `riscv-clang` 和 `riscv-clang++` 封装脚本。在调用时，脚本自动附加 `--target=riscv64-unknown-elf` 等 target triple 参数，并正确处理 `--version` 和 `--help` 选项。
- **T017-T018**: 运行测试脚本，验证生成的二进制文件是否为合法的 RISC-V ELF 格式（使用 `file` 命令验证）。

### 阶段 4: 用户故事 2 - 版本兼容性验证 (Phase 4: User Story 2)
- **T019-T021**: 在 `riscv-clang` 的 `--version` 中提取并显示 LLVM 的版本信息，并在测试脚本中添加版本查询与 ABI 兼容性相关的测试用例。
- **T022-T023**: 更新 `docs/docker-llvm-guide.md` 和 `specs/002-docker-llvm-toolchain/quickstart.md`，补充说明 Docker 工具链与 submodule 工具链版本的兼容性问题。

### 阶段 5: 用户故事 3 - 完善工具链体验 (Phase 5: User Story 3)
- **T024-T028**: 添加 `riscv-ld`, `riscv-objdump`, `riscv-strip` 的封装脚本。在测试脚本中增加关于无 Docker 权限、镜像未找到等场景下的错误提示测试。
- **T029-T034**: 在 `common.sh` 中增加对 `--docker-opts` 和 `--image` 参数的支持。提供友好的错误提示信息，完成 `docs/docker-llvm-guide.md` 使用手册的编写。

### 阶段 6: 完善与清理 (Phase 6: Polish & Cross-Cutting Concerns)
- **T035-T037**: 构建镜像并验证大小（目标 < 2GB），进行所有工具链的综合测试，并确保错误提示友好。
- **T038**: 运行完整的测试套件 `test-docker-llvm.sh`。
- **T039-T040**: 更新项目根目录的 `README.md` 以提及 Docker 工具链选项，验证 `quickstart.md` 指令的正确性。
- **T041**: 更新根目录下的 `.gitignore`，忽略测试产生的中间文件（如 `*.o`, `*.elf`，以及专门在 `tests/tools` 目录产生的测试产物）。

## 假设与决策 (Assumptions & Decisions)
- 假设宿主机平台为 Linux x86_64，且已安装并启动 Docker 服务。
- 决定采用 Debian bullseye 并从 `apt.llvm.org` 源安装 LLVM 13，因为这是最轻量且最贴近目标版本的方式，能保证镜像体积在 2GB 以内。
- `common.sh` 中的 `docker run` 命令将采用类似 `docker run --rm -v "$PWD:/work" -w /work --user $(id -u):$(id -g)` 的形式，以解决文件权限和路径上下文的问题。

## 验证步骤 (Verification steps)
1. 执行 `tests/tools/test-docker-llvm.sh`，检查所有测试用例是否通过。
2. 检查 Docker 镜像体积：`docker images | grep llvm` 确认是否小于 2GB。
3. 手动使用 `tools/docker-llvm/riscv-clang` 编译一段简单的 C 语言代码，并用 `file` 命令验证输出为 RISC-V ELF 文件。