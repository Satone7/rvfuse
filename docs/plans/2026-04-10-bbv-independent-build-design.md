# BBV 插件独立编译设计

## 背景

QEMU 子模块已从 xuantie-qemu 升级到官方 v9.2.4，原生包含 `bbv.c` 插件。之前在 xuantie-qemu 上通过 patches 添加了 `.disas` 输出和 exit flush 功能。为避免 fork QEMU 仓库和子模块 dirty 状态，采用**独立编译定制版 bbv.c** 方案。

## 目标

- QEMU 子模块保持干净（不修改任何文件）
- 定制版 bbv 插件（.disas + exit flush）独立编译为 `libbbv.so`
- 官方 QEMU 原生插件同时保留，可作为基线对比
- 更新项目中所有引用 bbv 插件路径的脚本

## 方案

### 1. 新建 `tools/bbv/bbv.c`

基于官方 `third_party/qemu/contrib/plugins/bbv.c`，合并 patch 0002 和 0003 的功能：

- **`.disas` 输出**：在 `vcpu_tb_trans` 中首次发现 BB 时，将汇编指令写入 `<base>.disas` 文件，格式为 `BB <id> (vaddr: <hex>, <N> insns):`
- **`plugin_flush()`**：在 `plugin_exit` 中调用，确保总指令数不足 interval 时仍能输出最后一批计数
- **保持 0-based BB 索引**（官方默认 `g_hash_table_size(bbs)`，不改）

### 2. 新建 `tools/bbv/Makefile`

```makefile
CC ?= gcc
CFLAGS := -shared -fPIC -O2 -Wall -Werror
CFLAGS += $(shell pkg-config --cflags glib-2.0)
LDFLAGS += $(shell pkg-config --libs glib-2.0)

QEMU_PLUGIN_HDR = ../../third_party/qemu/include/qemu/qemu-plugin.h

libbbv.so: bbv.c $(QEMU_PLUGIN_HDR)
	$(CC) $(CFLAGS) -o $@ bbv.c $(LDFLAGS)
```

- 输出文件名 `libbbv.so`（与官方一致）
- 依赖只读头文件 `qemu-plugin.h`，不修改子模块
- 独立于 QEMU 构建系统，`make -C tools/bbv/` 即可

### 3. 修改 `verify_bbv.sh`

- 在 `make plugins` 之后，追加 `make -C tools/bbv/` 编译定制版
- 同时验证官方插件和定制版插件均可构建
- 测试时使用定制版 `tools/bbv/libbbv.so`
- 验证插件路径更新为 `tools/bbv/libbbv.so`

### 4. 修改 `setup.sh`

- `STEP2_ARTIFACTS` 追加 `tools/bbv/libbbv.so`
- `step4_bbv_profiling()` 中 `plugin_so` 路径改为 `${PROJECT_ROOT}/tools/bbv/libbbv.so`
- 保留 `--enable-plugins`（官方插件仍构建）

### 5. 删除 `patches/qemu-bbv/`

不再需要，子模块保持纯净。

## 不影响的下游工具

`analyze_bbv.py`、`profile_to_dfg.sh`、`tools/dfg/` 均使用 `vaddr` 匹配 BB，不依赖 `bb_id` 基数，无需修改。

## 输出文件变更

| 之前（patch 版） | 之后（独立编译） |
|---|---|
| `third_party/qemu/build/contrib/plugins/bbv.so` | `tools/bbv/libbbv.so`（定制版） |
| 无官方插件使用 | `third_party/qemu/build/contrib/plugins/libbbv.so`（官方，保留） |
| `output/yolo.bbv.disas` | 同（由定制版 .disas 功能产生） |
