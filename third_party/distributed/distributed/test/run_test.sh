#! /bin/bash
set -e

cd "$(dirname $0)"
python3 ./test_distributed_wait.py --case correctness
