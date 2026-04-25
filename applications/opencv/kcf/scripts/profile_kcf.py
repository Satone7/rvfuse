#!/usr/bin/env python3
"""
KCF Perf Profiling Script
Profile KCF tracker on Banana Pi RISC-V hardware using Linux perf.
"""

import argparse
import os
import sys
import subprocess
import paramiko
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Profile KCF tracker on Banana Pi')
    parser.add_argument('--host', required=True, help='Banana Pi IP address')
    parser.add_argument('--user', default='root', help='SSH username')
    parser.add_argument('--password', required=True, help='SSH password')
    parser.add_argument('--runner', required=True, help='Local path to KCF binary')
    parser.add_argument('--libs', required=True, help='Local path to OpenCV libraries')
    parser.add_argument('--sysroot', required=True, help='Local path to sysroot')
    parser.add_argument('--test-data', required=True, help='Local path to test data directory')
    parser.add_argument('--remote-dir', default='/root/kcf-perf', help='Remote working directory')
    parser.add_argument('--outdir', default='output/perf/opencv-kcf', help='Local output directory')
    parser.add_argument('--freq', default=999, type=int, help='Sampling frequency (Hz)')
    parser.add_argument('--skip-upload', action='store_true', help='Skip upload (files already on board)')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without executing')
    return parser.parse_args()

def ssh_connect(host, user, password):
    """Connect to remote board via SSH."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    return client

def ssh_exec(client, cmd, dry_run=False):
    """Execute command on remote board."""
    if dry_run:
        print(f"[DRY-RUN] SSH: {cmd}")
        return "", "", 0
    stdin, stdout, stderr = client.exec_command(cmd)
    return stdout.read().decode(), stderr.read().decode(), stdout.channel.recv_exit_status()

def sftp_upload(client, local_path, remote_path, dry_run=False):
    """Upload file via SFTP."""
    if dry_run:
        print(f"[DRY-RUN] SFTP: {local_path} -> {remote_path}")
        return
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()

def sftp_upload_dir(client, local_dir, remote_dir, dry_run=False):
    """Upload directory recursively via SFTP."""
    if dry_run:
        print(f"[DRY-RUN] SFTP DIR: {local_dir} -> {remote_dir}")
        return

    sftp = client.open_sftp()
    try:
        sftp.mkdir(remote_dir)
    except:
        pass  # Directory may exist

    for item in Path(local_dir).iterdir():
        remote_item = f"{remote_dir}/{item.name}"
        if item.is_file():
            sftp.put(str(item), remote_item)
        elif item.is_dir():
            sftp_upload_dir(client, str(item), remote_item, dry_run)
    sftp.close()

def main():
    args = parse_args()

    # Resolve paths
    runner_path = Path(args.runner).resolve()
    libs_path = Path(args.libs).resolve()
    sysroot_path = Path(args.sysroot).resolve()
    test_data_path = Path(args.test_data).resolve()
    outdir_path = Path(args.outdir).resolve()

    # Verify local files exist
    if not runner_path.exists():
        print(f"Error: Runner not found: {runner_path}")
        sys.exit(1)
    if not libs_path.exists():
        print(f"Error: Libraries not found: {libs_path}")
        sys.exit(1)
    if not sysroot_path.exists():
        print(f"Error: Sysroot not found: {sysroot_path}")
        sys.exit(1)
    if not test_data_path.exists():
        print(f"Error: Test data not found: {test_data_path}")
        sys.exit(1)

    # Create output directory
    outdir_path.mkdir(parents=True, exist_ok=True)

    print(f"=== KCF Perf Profiling ===")
    print(f"Host: {args.host}")
    print(f"Runner: {runner_path}")
    print(f"Remote dir: {args.remote_dir}")

    # Connect to board
    print("\n[1] Connecting to Banana Pi...")
    if not args.dry_run:
        client = ssh_connect(args.host, args.user, args.password)
        print("Connected!")

    # Setup remote directory
    print("\n[2] Setting up remote directory...")
    ssh_exec(client, f"mkdir -p {args.remote_dir}/lib {args.remote_dir}/frames", args.dry_run)

    if not args.skip_upload:
        # Upload binary
        print("\n[3] Uploading KCF binary...")
        sftp_upload(client, str(runner_path), f"{args.remote_dir}/KCF", args.dry_run)

        # Upload libraries
        print("\n[4] Uploading OpenCV libraries...")
        for lib in libs_path.glob("libopencv*.so*"):
            sftp_upload(client, str(lib), f"{args.remote_dir}/lib/{lib.name}", args.dry_run)

        # Upload test data (frames, images.txt, region.txt)
        print("\n[5] Uploading test data...")
        sftp_upload_dir(client, str(test_data_path), f"{args.remote_dir}/frames", args.dry_run)

        # Check for images.txt and region.txt
        images_txt = test_data_path.parent / "images.txt"
        region_txt = test_data_path.parent / "region.txt"
        if images_txt.exists():
            sftp_upload(client, str(images_txt), f"{args.remote_dir}/images.txt", args.dry_run)
        if region_txt.exists():
            sftp_upload(client, str(region_txt), f"{args.remote_dir}/region.txt", args.dry_run)

    # Setup perf
    print("\n[6] Checking perf...")
    stdout, stderr, rc = ssh_exec(client, "perf --version", args.dry_run)
    if rc != 0 and not args.dry_run:
        print("Error: perf not installed on board")
        print("Install with: apt-get install linux-perf")
        sys.exit(1)

    # Lower perf_event_paranoid if needed
    ssh_exec(client, "sysctl -w kernel.perf_event_paranoid=0", args.dry_run)

    # Run perf stat
    print("\n[7] Running perf stat...")
    perf_stat_cmd = f"cd {args.remote_dir} && LD_LIBRARY_PATH={args.remote_dir}/lib perf stat -d -- ./KCF"
    stdout, stderr, rc = ssh_exec(client, perf_stat_cmd, args.dry_run)

    # Save perf stat results
    stat_file = outdir_path / "perf_stat.txt"
    stat_file.write_text(stdout + stderr)
    print(f"Saved: {stat_file}")

    # Run perf record
    print("\n[8] Running perf record...")
    perf_record_cmd = f"cd {args.remote_dir} && LD_LIBRARY_PATH={args.remote_dir}/lib perf record -e cpu-clock -g -F {args.freq} -o perf.data -- ./KCF"
    stdout, stderr, rc = ssh_exec(client, perf_record_cmd, args.dry_run)

    # Generate perf report
    print("\n[9] Generating perf report...")
    perf_report_cmd = f"cd {args.remote_dir} && perf report --stdio -n --percent-limit 0.5 -i perf.data"
    stdout, stderr, rc = ssh_exec(client, perf_report_cmd, args.dry_run)

    report_file = outdir_path / "perf_report.txt"
    report_file.write_text(stdout)
    print(f"Saved: {report_file}")

    # Generate perf annotate
    print("\n[10] Generating perf annotate...")
    perf_annotate_cmd = f"cd {args.remote_dir} && perf annotate --stdio -i perf.data --symfs {args.remote_dir}/lib"
    stdout, stderr, rc = ssh_exec(client, perf_annotate_cmd, args.dry_run)

    annotate_file = outdir_path / "perf_annotate.txt"
    annotate_file.write_text(stdout)
    print(f"Saved: {annotate_file}")

    # Download perf.data for local analysis
    print("\n[11] Downloading perf.data...")
    if not args.dry_run:
        sftp = client.open_sftp()
        sftp.get(f"{args.remote_dir}/perf.data", str(outdir_path / "perf.data"))
        sftp.close()
        print(f"Saved: {outdir_path / 'perf.data'}")

    # Cleanup
    print("\n[12] Cleaning up...")
    ssh_exec(client, f"rm -rf {args.remote_dir}/perf.data {args.remote_dir}/output.txt", args.dry_run)

    if not args.dry_run:
        client.close()

    print("\n=== Profiling Complete ===")
    print(f"Results saved to: {outdir_path}")
    print(f"  - perf_stat.txt: Global metrics")
    print(f"  - perf_report.txt: Function hotspots")
    print(f"  - perf_annotate.txt: Instruction hotspots")
    print(f"  - perf.data: Raw perf data (for local analysis)")

if __name__ == "__main__":
    main()