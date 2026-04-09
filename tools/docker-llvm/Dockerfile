FROM debian:bookworm-slim

# Install wget and gnupg for repository setup
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    && echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-22 main" >> /etc/apt/sources.list \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt-get update && apt-get install -y \
    clang-22 \
    lld-22 \
    llvm-22 \
    && ln -s /usr/bin/clang-22 /usr/bin/clang \
    && ln -s /usr/bin/clang++-22 /usr/bin/clang++ \
    && ln -s /usr/bin/lld-22 /usr/bin/lld \
    && ln -s /usr/bin/llvm-objdump-22 /usr/bin/llvm-objdump \
    && ln -s /usr/bin/llvm-strip-22 /usr/bin/llvm-strip \
    && ln -s /usr/bin/llvm-readelf-22 /usr/bin/llvm-readelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
