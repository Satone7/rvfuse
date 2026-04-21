#!/usr/bin/env python3
"""perf_scalar_profile.py — Remote perf profiling on RISC-V (Banana Pi) via SSH.

Uploads cross-compiled binaries, libraries, sysroot, and models to the board,
runs perf stat/record/annotate, downloads results back to the local machine.

Supports both ONNX Runtime (generic_ort_runner) and llama.cpp frameworks.
Auto-detects framework from model file extension:
  .onnx → generic_ort_runner:  ./runner <model.onnx> [iterations]
  .gguf → llama-cli:           ./runner -m <model.gguf> <extra_args>

Usage:
    # ONNX Runtime (generic_ort_runner) — use short names, auto-download
    python3 tools/perf_scalar_profile.py \\
        --host 192.168.1.22 --user root --password bianbu \\
        --remote-dir /root/ort-perf \\
        --runner output/cross-ort/generic_ort_runner \\
        --libs output/cross-ort/lib \\
        --sysroot output/cross-ort/sysroot \\
        --models resnet50 mobilenetv2 squeezenet \\
        --outdir output/perf/ort \\
        --iterations 30 --freq 999

    # llama.cpp — auto-download GGUF
    python3 tools/perf_scalar_profile.py \\
        --host 192.168.1.22 --user root --password bianbu \\
        --remote-dir /root/llama-perf \\
        --runner output/llama.cpp/bin/llama-cli \\
        --libs output/llama.cpp/lib \\
        --sysroot output/llama.cpp/sysroot \\
        --models qwen-0.5b-q4_0 \\
        --input "-p 'Hello world' -n 128" \\
        --outdir output/perf/llama

    # List available models
    python3 tools/perf_scalar_profile.py --list-models

    # Upload only
    python3 tools/perf_scalar_profile.py \\
        --host 192.168.1.22 --user root --password bianbu \\
        --upload-only --runner ... --models ...

    # Skip upload (files already on board)
    python3 tools/perf_scalar_profile.py \\
        --host 192.168.1.22 --user root --password bianbu \\
        --skip-upload --runner generic_ort_runner \\
        --models resnet50 --iterations 30
"""

import argparse
import os
import re
import sys
import stat
import urllib.request
import hashlib
from pathlib import Path

import paramiko


# ---------------------------------------------------------------------------
# Model registry — known models with download URLs
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    # --- ONNX models (vision classification) ---
    "resnet50": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx",
        "filename": "resnet50.onnx",
        "framework": "ort",
        "desc": "ResNet-50 v2 (98 MB, opset 7)",
    },
    "mobilenetv2": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx",
        "filename": "mobilenetv2.onnx",
        "framework": "ort",
        "desc": "MobileNetV2 (14 MB, opset 10)",
    },
    "squeezenet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "filename": "squeezenet.onnx",
        "framework": "ort",
        "desc": "SqueezeNet 1.1 (5 MB, opset 7)",
    },
    "shufflenet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx",
        "filename": "shufflenet.onnx",
        "framework": "ort",
        "desc": "ShuffleNet (9 MB, opset 9)",
    },
    "vgg16": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-7.onnx",
        "filename": "vgg16.onnx",
        "framework": "ort",
        "desc": "VGG-16 (528 MB, opset 7)",
    },
    "densenet121": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-121.onnx",
        "filename": "densenet121.onnx",
        "framework": "ort",
        "desc": "DenseNet-121 (opset 7)",
    },
    "inception": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx",
        "filename": "inception_v1.onnx",
        "framework": "ort",
        "desc": "Inception v1 (27 MB, opset 9)",
    },
    "efficientnet-lite4": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "filename": "efficientnet-lite4.onnx",
        "framework": "ort",
        "desc": "EfficientNet-Lite4 (opset 11)",
    },
    # --- GGUF models (LLM) ---
    "qwen-0.5b-q4_0": {
        "url": "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_0.gguf",
        "filename": "Qwen2.5-0.5B-Instruct-Q4_0.gguf",
        "framework": "llama",
        "desc": "Qwen2.5 0.5B Instruct Q4_0 (~350 MB)",
    },
    "qwen-1.5b-q4_0": {
        "url": "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_0.gguf",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_0.gguf",
        "framework": "llama",
        "desc": "Qwen2.5 1.5B Instruct Q4_0 (~1 GB)",
    },
}


def list_available_models():
    """Print all models in the registry."""
    print("Available models (short name → download):")
    print(f"  {'Name':<22s} {'Framework':<8s} {'Description'}")
    print(f"  {'─'*22} {'─'*8} {'─'*40}")
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:<22s} {info['framework']:<8s} {info['desc']}")
    print()
    print("You can also pass local file paths or URLs directly.")


def download_file(url: str, dest: str, desc: str = "") -> bool:
    """Download a file with progress display."""
    print(f"  [DL] {desc or os.path.basename(dest)}")
    print(f"       {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_dl_progress)
        print(f"\n  [OK] Saved to {dest}")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def _dl_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r       [{bar}] {pct:3d}% ({mb_done:.0f}/{mb_total:.0f} MB)")
    else:
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r       {mb_done:.0f} MB downloaded")
    sys.stdout.flush()


def resolve_model(model_arg: str, models_dir: str) -> str:
    """Resolve a model argument to a local file path.

    Resolution order:
    1. Exact local path (if file exists)
    2. models_dir/filename (if file exists)
    3. Registry short name → download to models_dir
    4. URL → download to models_dir
    """
    # 1. Exact local path
    if os.path.isfile(model_arg):
        return os.path.abspath(model_arg)

    # 2. Check models_dir with the basename
    basename = os.path.basename(model_arg)
    candidate = os.path.join(models_dir, basename)
    if os.path.isfile(candidate):
        return candidate

    # 3. Registry short name (case-insensitive)
    key = model_arg.lower().strip()
    if key in MODEL_REGISTRY:
        info = MODEL_REGISTRY[key]
        dest = os.path.join(models_dir, info["filename"])
        if os.path.isfile(dest):
            print(f"  [OK] Model '{key}' already downloaded: {dest}")
            return dest
        os.makedirs(models_dir, exist_ok=True)
        if download_file(info["url"], dest, info["desc"]):
            return dest
        print(f"  [ERROR] Failed to download model '{key}'")
        sys.exit(1)

    # 4. URL (starts with http:// or https://)
    if model_arg.startswith("http://") or model_arg.startswith("https://"):
        filename = basename or "downloaded_model"
        if not filename.endswith((".onnx", ".ort", ".gguf")):
            filename += ".bin"
        dest = os.path.join(models_dir, filename)
        if os.path.isfile(dest):
            print(f"  [OK] Already downloaded: {dest}")
            return dest
        os.makedirs(models_dir, exist_ok=True)
        if download_file(model_arg, dest, f"Downloading {filename}"):
            return dest
        sys.exit(1)

    # 5. Not found
    print(f"  [ERROR] Model not found: {model_arg}")
    print(f"         Checked: {model_arg}, {candidate}")
    print(f"         Not in registry. Available: {', '.join(MODEL_REGISTRY.keys())}")
    print(f"         Or pass a URL / local file path.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Framework detection and command building
# ---------------------------------------------------------------------------

def detect_framework(model_name: str) -> str:
    """Auto-detect framework from model file extension."""
    if model_name.endswith(".onnx") or model_name.endswith(".ort"):
        return "ort"
    elif model_name.endswith(".gguf"):
        return "llama"
    else:
        return "unknown"


def build_run_cmd(runner_path: str, model_name: str, framework: str,
                  iterations: int, input_arg: str) -> str:
    """Build the runner command line (without perf prefix)."""
    if framework == "ort":
        # generic_ort_runner <model.onnx> [iterations]
        return f"{runner_path} {model_name} {iterations}"
    elif framework == "llama":
        # llama-cli -m <model.gguf> <extra_args>
        return f"{runner_path} -m {model_name} {input_arg}"
    else:
        # Fallback: pass everything through
        return f"{runner_path} {model_name} {input_arg}"


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def ssh_connect(host: str, user: str, password: str, port: int = 22) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, port=port, timeout=15)
    return ssh


def ssh_exec(ssh: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[str, str, int]:
    print(f"  [SSH] {cmd[:120]}{'...' if len(cmd)>120 else ''}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    return out, err, exit_code


def sftp_upload(ssh: paramiko.SSHClient, local: str, remote: str) -> None:
    sftp = ssh.open_sftp()
    if os.path.isfile(local):
        remote_dir = os.path.dirname(remote)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            _sftp_mkdir_p(sftp, remote_dir)
        print(f"  [SFTP] {local} -> {remote}")
        sftp.put(local, remote)
        st = os.stat(local)
        if st.st_mode & stat.S_IXUSR:
            try:
                sftp.chmod(remote, st.st_mode & 0o777)
            except IOError:
                pass
    elif os.path.isdir(local):
        for root, dirs, files in os.walk(local):
            rel_root = os.path.relpath(root, local)
            for f in files:
                local_file = os.path.join(root, f)
                remote_file = os.path.join(remote, rel_root, f) if rel_root != "." else os.path.join(remote, f)
                sftp_upload_file(sftp, local_file, remote_file)
    else:
        print(f"  [WARN] Skipping {local}: not found")
    sftp.close()


def sftp_upload_file(sftp, local: str, remote: str) -> None:
    remote_dir = os.path.dirname(remote)
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        _sftp_mkdir_p(sftp, remote_dir)
    print(f"  [SFTP] {local} -> {remote}")
    sftp.put(local, remote)
    st = os.stat(local)
    if st.st_mode & stat.S_IXUSR:
        try:
            sftp.chmod(remote, st.st_mode & 0o777)
        except IOError:
            pass


def _sftp_mkdir_p(sftp, path):
    parts = path.split("/")
    current = ""
    for part in parts:
        if not part:
            continue
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def sftp_download(ssh: paramiko.SSHClient, remote: str, local: str) -> bool:
    sftp = ssh.open_sftp()
    try:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        print(f"  [SFTP] {remote} -> {local}")
        sftp.get(remote, local)
        sftp.close()
        return True
    except FileNotFoundError:
        print(f"  [WARN] Remote file not found: {remote}")
        sftp.close()
        return False


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_workload(ssh: paramiko.SSHClient, remote_dir: str,
                    runner: str, models: list[str],
                    libs_dir: str | None, sysroot: str | None) -> None:
    print("\n=== Uploading workload ===")
    ssh_exec(ssh, f"mkdir -p {remote_dir}/lib")

    # Runner
    runner_name = os.path.basename(runner)
    sftp_upload(ssh, runner, f"{remote_dir}/{runner_name}")

    # Shared libraries
    if libs_dir and os.path.isdir(libs_dir):
        for f in os.listdir(libs_dir):
            if f.endswith(".so") or ".so." in f:
                sftp_upload(ssh, os.path.join(libs_dir, f), f"{remote_dir}/lib/{f}")

    # Sysroot
    if sysroot and os.path.isdir(sysroot):
        print(f"  [SFTP] Uploading sysroot: {sysroot} -> {remote_dir}/sysroot")
        sftp_upload(ssh, sysroot, f"{remote_dir}/sysroot")

    # Models
    for model in models:
        sftp_upload(ssh, model, f"{remote_dir}/{os.path.basename(model)}")

    print("  [OK] Upload complete")


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def check_remote_env(ssh: paramiko.SSHClient) -> None:
    out, err, rc = ssh_exec(ssh, "uname -m && perf --version 2>&1 && df -h / | tail -1")
    if rc != 0:
        print(f"  [ERROR] Remote check failed: {err}")
        sys.exit(1)
    print(f"  [OK] {out.strip()}")

    # Ensure perf can sample
    out2, _, _ = ssh_exec(ssh, "cat /proc/sys/kernel/perf_event_paranoid")
    paranoid = out2.strip()
    if paranoid and int(paranoid) > 1:
        print(f"  [WARN] perf_event_paranoid={paranoid}, setting to 0 for sampling...")
        ssh_exec(ssh, "echo 0 > /proc/sys/kernel/perf_event_paranoid")


def profile_model(ssh: paramiko.SSHClient, remote_dir: str,
                  runner_name: str, model_name: str,
                  framework: str, iterations: int, input_arg: str,
                  freq: int, has_libs: bool) -> dict:
    print(f"\n--- Profiling: {model_name} [{framework}] ---")

    work_dir = f"{remote_dir}/perf_{model_name.replace('.onnx','').replace('.gguf','')}"
    ssh_exec(ssh, f"mkdir -p {work_dir}")

    # Build LD_LIBRARY_PATH
    ld_paths = []
    if has_libs:
        ld_paths.append(f"{remote_dir}/lib")
    ld_paths.append(f"{remote_dir}/sysroot/lib")
    ld_paths.append(f"{remote_dir}/sysroot/lib/riscv64-linux-gnu")
    ld_prefix = f"LD_LIBRARY_PATH={':'.join(ld_paths)}:$LD_LIBRARY_PATH "

    runner_path = f"{remote_dir}/{runner_name}"

    # Build the runner command for this framework
    run_cmd = build_run_cmd(runner_path, model_name, framework, iterations, input_arg)

    # 1. perf stat
    print("  [1/3] perf stat ...")
    stat_out = f"{work_dir}/perf_stat.txt"
    cmd = f"{ld_prefix}perf stat -d -o {stat_out} -- {run_cmd}"
    _, err, rc = ssh_exec(ssh, f"cd {remote_dir} && {cmd}", timeout=600)

    # 2. perf record (use cpu-clock: RISC-V SBI PMU hardware sampling often fails)
    print("  [2/3] perf record ...")
    data_file = f"{work_dir}/perf.data"
    cmd = f"{ld_prefix}perf record -e cpu-clock -g -F {freq} -o {data_file} -- {run_cmd}"
    _, err, rc = ssh_exec(ssh, f"cd {remote_dir} && {cmd}", timeout=600)

    # 3. Reports
    print("  [3/3] Generating reports ...")
    report_out = f"{work_dir}/perf_report.txt"
    annotate_out = f"{work_dir}/perf_annotate.txt"
    ssh_exec(ssh, f"cd {remote_dir} && perf report --stdio -n --percent-limit 0.5 -i {data_file} > {report_out} 2>/dev/null", timeout=300)
    ssh_exec(ssh, f"cd {remote_dir} && perf annotate --stdio -i {data_file} > {annotate_out} 2>/dev/null", timeout=300)

    # Extract metrics
    metrics = {"name": model_name.replace(".onnx", "").replace(".gguf", ""),
               "framework": framework}
    stat_content, _, _ = ssh_exec(ssh, f"cat {stat_out}")
    if stat_content:
        for line in stat_content.splitlines():
            line = line.strip()
            m = re.match(r"([\d,]+)\s+cycles", line)
            if m and "cycles" not in metrics:
                metrics["cycles"] = m.group(1).replace(",", "")
            m = re.match(r"([\d,]+)\s+instructions", line)
            if m and "instructions" not in metrics:
                metrics["instructions"] = m.group(1).replace(",", "")
            m = re.match(r"([\d.]+)\s+insn per cycle", line)
            if m:
                metrics["ipc"] = m.group(1)
            m = re.match(r"([\d.]+)%\s+of all cache refs", line)
            if m:
                metrics["cache_miss_pct"] = m.group(1)

    report_content, _, _ = ssh_exec(ssh, f"head -60 {report_out}")
    if report_content:
        for line in report_content.splitlines():
            # perf report --stdio format: "  Children%  Self%  Samples  Cmd  DSO  [.] Symbol"
            # Match lines with [.] symbol marker to get function name
            m = re.match(r"\s+([\d.]+)%\s+[\d.]+%\s+\d+\s+\S+\s+\S+\s+\[\.\]\s+(.+)", line)
            if m:
                metrics["top_function"] = m.group(2).strip()
                metrics["top_pct"] = m.group(1)
                break

    return metrics


def download_results(ssh: paramiko.SSHClient, remote_dir: str,
                     model_name: str, local_dir: str) -> None:
    ext = ".onnx" if model_name.endswith(".onnx") else ".gguf" if model_name.endswith(".gguf") else ""
    work_dir = f"{remote_dir}/perf_{model_name.replace(ext, '')}"
    local_model_dir = os.path.join(local_dir, model_name.replace(ext, ""))
    os.makedirs(local_model_dir, exist_ok=True)

    for f in ["perf_stat.txt", "perf_report.txt", "perf_annotate.txt"]:
        sftp_download(ssh, f"{work_dir}/{f}", os.path.join(local_model_dir, f))


def cleanup_remote(ssh: paramiko.SSHClient, remote_dir: str, model_name: str) -> None:
    ext = ".onnx" if model_name.endswith(".onnx") else ".gguf" if model_name.endswith(".gguf") else ""
    work_dir = f"{remote_dir}/perf_{model_name.replace(ext, '')}"
    ssh_exec(ssh, f"rm -rf {work_dir}")


def generate_summary(all_metrics: list[dict], outdir: str, freq: int,
                     host: str, runner: str) -> None:
    summary_path = os.path.join(outdir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Perf Profiling Summary\n\n")
        f.write(f"Generated: {os.popen('date').read().strip()}\n")
        f.write(f"Host: {host}\n")
        f.write(f"Runner: {runner}\n")
        f.write(f"Sampling frequency: {freq} Hz (cpu-clock)\n\n")

        f.write("## Overview\n\n")
        f.write("| Model | Framework | Cycles | Instructions | IPC | Cache Miss % | Top Function | Top % |\n")
        f.write("|-------|-----------|--------|-------------|-----|-------------|-------------|-------|\n")
        for m in all_metrics:
            f.write(
                f"| {m.get('name', '?')} "
                f"| {m.get('framework', '?')} "
                f"| {m.get('cycles', 'N/A')} "
                f"| {m.get('instructions', 'N/A')} "
                f"| {m.get('ipc', 'N/A')} "
                f"| {m.get('cache_miss_pct', 'N/A')} "
                f"| {m.get('top_function', 'N/A')} "
                f"| {m.get('top_pct', 'N/A')} |\n"
            )
        f.write("\n")

        f.write("## Top Functions per Model\n\n")
        for m in all_metrics:
            name = m.get("name", "?")
            report_path = os.path.join(outdir, name, "perf_report.txt")
            if os.path.isfile(report_path):
                f.write(f"### {name}\n\n```\n")
                count = 0
                with open(report_path) as rf:
                    for line in rf:
                        if re.match(r"\s+[\d.]+%", line):
                            f.write(f"  {line.strip()}\n")
                            count += 1
                            if count >= 10:
                                break
                f.write("```\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Review `perf_annotate.txt` for instruction-level hotspots\n")
        f.write("2. Look for fusion patterns: load-compute-store, MAC chains, address calculations\n")
        f.write("3. Shared hot functions across models → highest-priority fusion targets\n")

    print(f"\nSummary written to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remote perf profiling on RISC-V Banana Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", help="Banana Pi IP")
    parser.add_argument("--user", default="root", help="SSH username")
    parser.add_argument("--password", help="SSH password")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--runner", help="Local path to inference binary")
    parser.add_argument("--models", nargs="+",
                        help="Model files: local paths, short names (resnet50, mobilenetv2, "
                             "qwen-0.5b-q4_0), or URLs. Auto-downloads if not found.")
    parser.add_argument("--models-dir", default="output/models",
                        help="Local directory for downloaded models (default: output/models)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--input", default="",
                        help="Extra args for runner (llama.cpp: '-p \"Hello\" -n 128'). "
                             "Not needed for ORT generic runner.")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Inference iterations for ORT runner (default: 30)")
    parser.add_argument("--libs", default=None,
                        help="Local directory with shared libraries (.so)")
    parser.add_argument("--sysroot", default=None, help="Local sysroot directory")
    parser.add_argument("--outdir", default="output/perf", help="Local output directory")
    parser.add_argument("--remote-dir", default="/root", help="Remote working directory")
    parser.add_argument("--freq", type=int, default=999, help="perf sampling frequency (Hz)")
    parser.add_argument("--upload-only", action="store_true", help="Upload only, skip profiling")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")

    # Use parse_known_args so --list-models works without required args
    args, _ = parser.parse_known_args()

    if args.list_models:
        list_available_models()
        return

    # Manual required-arg validation (skipped for --list-models)
    missing = []
    if not args.host: missing.append("--host")
    if not args.password: missing.append("--password")
    if not args.runner: missing.append("--runner")
    if not args.models: missing.append("--models")
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    # Resolve model paths: local file → models_dir → registry download → URL download
    resolved_models = []
    for m in args.models:
        resolved = resolve_model(m, args.models_dir)
        resolved_models.append(resolved)
        if resolved != m:
            print(f"  [MODEL] {m} → {resolved}")
    args.models = resolved_models

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"  Host: {args.user}@{args.host}:{args.port}")
        print(f"  Remote dir: {args.remote_dir}")
        print(f"  Runner: {args.runner}")
        print(f"  Models: {args.models}")
        print(f"  Input args: {args.input or '(none)'}")
        print(f"  Iterations: {args.iterations}")
        print(f"  Libs: {args.libs}")
        print(f"  Sysroot: {args.sysroot}")
        print(f"  Outdir: {args.outdir}")
        print(f"  Freq: {args.freq}")
        if not args.skip_upload:
            print(f"    - Upload {args.runner} -> {args.remote_dir}/{os.path.basename(args.runner)}")
            if args.libs:
                print(f"    - Upload libs {args.libs} -> {args.remote_dir}/lib/")
            if args.sysroot:
                print(f"    - Upload sysroot {args.sysroot} -> {args.remote_dir}/sysroot/")
            for m in args.models:
                print(f"    - Upload {m} -> {args.remote_dir}/{os.path.basename(m)}")
        if not args.upload_only:
            for m in args.models:
                name = os.path.basename(m)
                fw = detect_framework(name)
                run_cmd = build_run_cmd(f"{args.remote_dir}/{os.path.basename(args.runner)}",
                                        name, fw, args.iterations, args.input)
                print(f"    - Profile {name} [{fw}]: {run_cmd}")
                print(f"    - Download -> {args.outdir}/{name}/")
            print(f"    - Generate {args.outdir}/summary.md")
        return

    # Connect
    print(f"=== Connecting to {args.user}@{args.host} ===")
    ssh = ssh_connect(args.host, args.user, args.password, args.port)
    check_remote_env(ssh)

    # Upload
    has_libs = (args.libs and os.path.isdir(args.libs)) or (args.sysroot and os.path.isdir(args.sysroot))
    if not args.skip_upload:
        upload_workload(ssh, args.remote_dir, args.runner, args.models,
                        args.libs, args.sysroot)
    else:
        print("\n  [SKIP] Upload skipped (--skip-upload)")

    if args.upload_only:
        print("\n=== Upload complete (upload-only mode) ===")
        ssh.close()
        return

    # Profile
    os.makedirs(args.outdir, exist_ok=True)
    all_metrics = []
    runner_name = os.path.basename(args.runner)

    for model_path in args.models:
        model_name = os.path.basename(model_path)
        framework = detect_framework(model_name)
        metrics = profile_model(ssh, args.remote_dir, runner_name,
                                model_name, framework,
                                args.iterations, args.input,
                                args.freq, has_libs)
        download_results(ssh, args.remote_dir, model_name, args.outdir)
        cleanup_remote(ssh, args.remote_dir, model_name)
        all_metrics.append(metrics)

    generate_summary(all_metrics, args.outdir, args.freq, args.host, args.runner)

    ssh.close()
    print(f"\n=== Profiling complete ===")
    print(f"Results in: {args.outdir}/")


if __name__ == "__main__":
    main()
