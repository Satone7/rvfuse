FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
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
        && rm -rf /var/lib/apt/lists/

# Work directories
WORKDIR /build

# ORT source, Eigen, LLVM toolchain, sysroot, and YOLO runner are
# mounted via volumes at runtime (see build.sh).
