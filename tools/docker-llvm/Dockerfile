FROM debian:bullseye-slim

# Install wget and gnupg for repository setup
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && echo "deb http://apt.llvm.org/bullseye/ llvm-toolchain-bullseye-13 main" >> /etc/apt/sources.list \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt-get update && apt-get install -y \
    clang-13 \
    lld-13 \
    llvm-13 \
    && ln -s /usr/bin/clang-13 /usr/bin/clang \
    && ln -s /usr/bin/clang++-13 /usr/bin/clang++ \
    && ln -s /usr/bin/lld-13 /usr/bin/lld \
    && ln -s /usr/bin/llvm-objdump-13 /usr/bin/llvm-objdump \
    && ln -s /usr/bin/llvm-strip-13 /usr/bin/llvm-strip \
    && ln -s /usr/bin/llvm-readelf-13 /usr/bin/llvm-readelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
