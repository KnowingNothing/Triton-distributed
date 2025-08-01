#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

pushd ${PROJECT_ROOT}

rm python/setup.cfg
echo "" >> python/setup.cfg
echo "[build_ext]" >> python/setup.cfg
echo "base-dir=/project" >> python/setup.cfg

export CIBW_ENVIRONMENT="MAX_JOBS=128 \
        TRITON_BUILD_WITH_CLANG_LLD=1\
        http_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
        https_proxy=http://sys-proxy-rd-relay.byted.org:8118"

# many_linux_2_28 image comes with GCC 12.2.1, but not clang.
# With this install, it gets clang 16.0.6.
export CIBW_BEFORE_ALL="dnf install clang lld -y"

export CIBW_MANYLINUX_X86_64_IMAGE="quay.io/pypa/manylinux_2_28_x86_64:latest"

export CIBW_BUILD="cp3{9,10,11,12,13,13t}-manylinux_x86_64"
export CIBW_SKIP="cp{35,36,37,38}-*"
export CIBW_FREE_THREADED_SUPPORT=1
python3 -m cibuildwheel python --output-dir wheelhouse

popd
