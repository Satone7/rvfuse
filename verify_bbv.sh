#!/bin/bash
set -e

# ==============================================================================
# QEMU BBV Plugin Verification Script
# 此脚本用于一键验证 QEMU BBV (Basic Block Vector) 插件的编译和运行流程。
# ==============================================================================

WORKSPACE="/workspace"
QEMU_DIR="${WORKSPACE}/third_party/qemu"
DEMO_SRC="${WORKSPACE}/demo.c"
DEMO_ELF="${WORKSPACE}/demo.elf"
BBV_OUT="${WORKSPACE}/bbv.out"
QEMU_BIN="${QEMU_DIR}/build/qemu-riscv64"
PLUGIN_SO="${QEMU_DIR}/build/contrib/plugins/libbbv.so"

echo "========================================"
echo "1. 编译 Demo 测试程序"
echo "========================================"
if [ ! -f "${DEMO_SRC}" ]; then
    echo "未找到 demo.c，正在创建..."
    cat << 'EOF' > "${DEMO_SRC}"
void _start() {
    int sum = 0;
    for (int i = 0; i < 1000; i++) {
        if (i % 2 == 0) {
            sum += i;
        } else {
            sum -= i;
        }
    }
    
    // exit syscall
    asm volatile(
        "li a7, 93\n\t" // sys_exit
        "li a0, 0\n\t"  // status 0
        "ecall\n\t"
    );
}
EOF
fi

echo "使用 docker-llvm/riscv-clang 编译 demo.c ..."
${WORKSPACE}/tools/docker-llvm/riscv-clang -o "${DEMO_ELF}" "${DEMO_SRC}"
if [ -f "${DEMO_ELF}" ]; then
    echo "[OK] Demo 编译成功: ${DEMO_ELF}"
else
    echo "[ERROR] Demo 编译失败!"
    exit 1
fi
echo ""

echo "========================================"
echo "2. 编译 QEMU 及 BBV 插件"
echo "========================================"
cd "${QEMU_DIR}"

if [ ! -f "${QEMU_BIN}" ] || [ ! -f "${PLUGIN_SO}" ]; then
    echo "正在配置和编译 QEMU (riscv64-linux-user) ..."
    ./configure --target-list=riscv64-linux-user --disable-werror
    
    echo "正在编译 QEMU 主程序 ..."
    make -j$(nproc)
    
    echo "正在编译 QEMU 插件 ..."
    make plugins
else
    echo "QEMU 和插件已存在，跳过编译步骤以节省时间。"
    echo "如果需要重新编译，请先删除 ${QEMU_BIN} 或 ${PLUGIN_SO}"
fi

if [ -f "${QEMU_BIN}" ] && [ -f "${PLUGIN_SO}" ]; then
    echo "[OK] QEMU 及 BBV 插件编译成功。"
else
    echo "[ERROR] QEMU 或插件编译失败!"
    exit 1
fi
echo ""

echo "========================================"
echo "3. 运行并验证 BBV 插件"
echo "========================================"
cd "${WORKSPACE}"

# 清理旧的输出文件
rm -f ${BBV_OUT}*

echo "执行命令:"
echo "${QEMU_BIN} -plugin ${PLUGIN_SO},interval=100,outfile=${BBV_OUT} ${DEMO_ELF}"
${QEMU_BIN} -plugin ${PLUGIN_SO},interval=100,outfile=${BBV_OUT} ${DEMO_ELF}

if [ -f "${BBV_OUT}.0.bb" ]; then
    echo "[OK] 成功生成 BBV 输出文件: ${BBV_OUT}.0.bb"
    echo "----------------------------------------"
    echo "文件内容片段 (前5行):"
    head -n 5 "${BBV_OUT}.0.bb"
    echo "----------------------------------------"
    echo "验证流程全部成功完成！"
else
    echo "[ERROR] 未能生成预期的 BBV 输出文件!"
    exit 1
fi
