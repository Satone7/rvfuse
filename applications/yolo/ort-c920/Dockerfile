FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        gpg wget software-properties-common \
        && rm -rf /var/lib/apt/lists/ \
    && wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        git \
        ca-certificates \
        make \
        libprotobuf-dev \
        protobuf-compiler \
        libflatbuffers-dev \
        gcc-riscv64-linux-gnu \
        binutils-riscv64-linux-gnu \
        && rm -rf /var/lib/apt/lists/ \
    && mkdir -p /riscv64-gcc-bin \
    && ln -sf /usr/bin/riscv64-linux-gnu-ld /riscv64-gcc-bin/ld \
    && ln -sf /usr/bin/riscv64-linux-gnu-as /riscv64-gcc-bin/as

# Work directories
WORKDIR /build

# ORT source, Eigen, LLVM toolchain, sysroot, and YOLO runner are
# mounted via volumes at runtime (see build.sh).
