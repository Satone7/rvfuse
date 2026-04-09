#!/bin/bash
# Shared functions for Docker LLVM toolchain

# Default configurations
DEFAULT_IMAGE="rvfuse/llvm-riscv:22"
DOCKER_IMAGE="${RVFUSE_LLVM_IMAGE:-$DEFAULT_IMAGE}"
DOCKER_OPTS=""

# ANSI color codes for error messages
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# T007: Error message functions
error_msg() {
    echo -e "${RED}Error: $1${NC}" >&2
}

warn_msg() {
    echo -e "${YELLOW}Warning: $1${NC}" >&2
}

# T008: Environment variable and custom args handling
# This parses out --image and --docker-opts from the arguments
parse_docker_args() {
    local NEW_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --image=*)
                DOCKER_IMAGE="${1#*=}"
                shift
                ;;
            --docker-opts=*)
                DOCKER_OPTS="${1#*=}"
                shift
                ;;
            *)
                NEW_ARGS+=("$1")
                shift
                ;;
        esac
    done
    # Export parsed args back
    COMMAND_ARGS=("${NEW_ARGS[@]}")
}

# T005: Docker availability check
check_docker() {
    if ! command -v docker &> /dev/null; then
        error_msg "Docker is not installed or not in PATH."
        echo "Please install Docker to use this toolchain." >&2
        echo "See: https://docs.docker.com/get-docker/" >&2
        exit 1
    fi

    if ! docker info &> /dev/null; then
        error_msg "Docker daemon is not running or you do not have permission to access it."
        echo "Try running 'sudo systemctl start docker' or adding your user to the 'docker' group." >&2
        exit 1
    fi
}

# T006: Image pull/build function
ensure_image() {
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        warn_msg "Docker image '$DOCKER_IMAGE' not found locally."
        
        # If it's the default image, we can try to build it automatically
        if [[ "$DOCKER_IMAGE" == "$DEFAULT_IMAGE" ]]; then
            echo "Building default image from tools/docker-llvm/Dockerfile..." >&2
            local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
            if ! docker build -t "$DOCKER_IMAGE" "$SCRIPT_DIR" >&2; then
                error_msg "Failed to build Docker image."
                exit 1
            fi
        else
            echo "Attempting to pull image '$DOCKER_IMAGE'..." >&2
            if ! docker pull "$DOCKER_IMAGE" >&2; then
                error_msg "Failed to pull Docker image '$DOCKER_IMAGE'. Please check your network connection or the image name."
                exit 1
            fi
        fi
    fi
}

# T009: Docker run wrapper function
run_in_docker() {
    local cmd="$1"
    shift

    check_docker
    parse_docker_args "$@"
    ensure_image

    # Use interactive mode if terminal is a tty
    local IT_FLAG=""
    if [ -t 0 ]; then
        IT_FLAG="-it"
    fi

    # Execute command inside docker
    # -v "$PWD:/work" mounts current directory
    # -w /work sets working directory
    # --user maps to current user to avoid root-owned files
    docker run --rm $IT_FLAG \
        -v "$PWD:/work" \
        -w /work \
        --user $(id -u):$(id -g) \
        $DOCKER_OPTS \
        "$DOCKER_IMAGE" \
        "$cmd" "${COMMAND_ARGS[@]}"
}
