#!/usr/bin/env python3
"""
BBV 分析脚本：计算基本块执行次数并排序
用法: python3 analyze_bbv.py <bb_file> <disas_file>
"""

import re
import sys
from collections import defaultdict


def parse_disas(disas_file):
    """从 .disas 文件解析每个 BB 的指令数"""
    bb_insns = {}
    bb_vaddr = {}

    with open(disas_file, 'r') as f:
        for line in f:
            # 匹配: BB 3 (vaddr: 0x1113a, 3 insns):
            m = re.match(r'BB (\d+) \(vaddr: (0x[0-9a-f]+), (\d+) insns\)', line)
            if m:
                bb_num = int(m.group(1))
                bb_insns[bb_num] = int(m.group(3))
                bb_vaddr[bb_num] = m.group(2)

    return bb_insns, bb_vaddr


def parse_bb(bb_file):
    """从 .bb 文件解析每个 BB 的累计指令执行数"""
    bb_total_insns = defaultdict(int)
    interval_count = 0

    with open(bb_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('T'):
                interval_count += 1
                # 格式: T:3:12 :7:10 :9:16 ...
                # 移除开头的 'T'，然后按空格分割
                content = line[1:].strip()
                parts = content.split()
                for part in parts:
                    # 格式可能是 "3:12" 或 ":3:12" (首个元素)
                    part = part.lstrip(':')
                    if ':' in part:
                        bb_num, insns = part.split(':')
                        bb_total_insns[int(bb_num)] += int(insns)

    return bb_total_insns, interval_count


def analyze(bb_file, disas_file):
    """分析并输出 BB 执行次数排序"""
    bb_insns, bb_vaddr = parse_disas(disas_file)
    bb_total_insns, interval_count = parse_bb(bb_file)

    # 计算执行次数
    results = []
    for bb_num in sorted(bb_total_insns.keys()):
        total_insns = bb_total_insns[bb_num]
        insns_per_exec = bb_insns.get(bb_num, 1)
        exec_count = total_insns // insns_per_exec
        vaddr = bb_vaddr.get(bb_num, 'unknown')
        results.append({
            'bb': bb_num,
            'vaddr': vaddr,
            'exec_count': exec_count,
            'total_insns': total_insns,
            'insns_per_exec': insns_per_exec
        })

    # 按执行次数降序排序
    results.sort(key=lambda x: -x['exec_count'])

    # 输出结果
    print("=" * 70)
    print("基本块执行次数排序分析")
    print("=" * 70)
    print(f"输入文件: {bb_file}, {disas_file}")
    print(f"时间间隔数: {interval_count}")
    print(f"基本块总数: {len(results)}")
    print("=" * 70)
    print()

    # 表头
    print(f"{'排名':<6} {'BB#':<6} {'地址':<12} {'执行次数':<12} {'累计指令':<12} {'BB指令数':<10}")
    print("-" * 70)

    # 表格内容
    for rank, r in enumerate(results, 1):
        print(f"{rank:<6} {r['bb']:<6} {r['vaddr']:<12} {r['exec_count']:<12} {r['total_insns']:<12} {r['insns_per_exec']:<10}")

    print()
    print("=" * 70)
    print("热点基本块 (执行次数 Top 5):")
    print("=" * 70)
    for rank, r in enumerate(results[:5], 1):
        print(f"#{rank} BB {r['bb']} @ {r['vaddr']}")
        print(f"   执行次数: {r['exec_count']}, 累计指令: {r['total_insns']}, BB大小: {r['insns_per_exec']} 条指令")
        print()

    return results


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"用法: {sys.argv[0]} <bb_file> <disas_file>")
        print(f"示例: {sys.argv[0]} bbv.out.0.bb bbv.out.disas")
        sys.exit(1)

    bb_file = sys.argv[1]
    disas_file = sys.argv[2]

    try:
        analyze(bb_file, disas_file)
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}")
        sys.exit(1)