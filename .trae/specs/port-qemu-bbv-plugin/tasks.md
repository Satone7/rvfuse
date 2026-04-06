# Tasks
- [x] Task 1: 移植并修改 bbv.c 插件代码
  - [x] SubTask 1.1: 拷贝 `/workspace/bbv_upstream.c` 到 `/workspace/third_party/qemu/contrib/plugins/bbv.c`。
  - [x] SubTask 1.2: 对 `bbv.c` 源码进行 Patch，将 `qemu_plugin_register_vcpu_tb_exec_cond_cb` 替换为 `qemu_plugin_register_vcpu_tb_exec_cb`，并在 `vcpu_interval_exec` 中加入 `if (vcpu->count < interval) return;` 条件判断。
  - [x] SubTask 1.3: 修改 `/workspace/third_party/qemu/contrib/plugins/Makefile`，将 `bbv` 加入到 `NAMES` 列表中以支持编译。

- [x] Task 2: 编译 QEMU 和 Plugins
  - [x] SubTask 2.1: 在 `/workspace/third_party/qemu` 目录下执行 `./configure --target-list=riscv64-linux-user` 开启配置。
  - [x] SubTask 2.2: 运行 `make -j$(nproc)` 编译 qemu-riscv64。
  - [x] SubTask 2.3: 运行 `make plugins` 确保生成 `libbbv.so`（位于 `build/contrib/plugins/` 或 `contrib/plugins/` 下）。

- [x] Task 3: 创建并编译 Demo 测试程序
  - [x] SubTask 3.1: 在 `/workspace` 下创建一个 `demo.c`，包含一些循环和基本运算。
  - [x] SubTask 3.2: 使用 `/workspace/tools/docker-llvm/riscv-clang` 脚本编译 `demo.c` 生成 `demo.elf` 静态/动态可执行文件。

- [x] Task 4: 运行并验证插件
  - [x] SubTask 4.1: 使用编译好的 `qemu-riscv64` 运行 `demo.elf`，并通过 `-plugin` 参数挂载 `libbbv.so`。命令示例：`qemu-riscv64 -plugin ./libbbv.so,interval=100,outfile=bbv.out demo.elf`。
  - [x] SubTask 4.2: 检查当前目录是否成功生成了 `bbv.out.0.bb` 文件，并且文件中包含基本的向量数据记录（如以 `T` 开头的行）。

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 4] depends on [Task 2]
- [Task 4] depends on [Task 3]
