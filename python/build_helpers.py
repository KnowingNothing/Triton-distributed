################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import os
import sysconfig
import sys
from pathlib import Path


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "python" / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def create_symlink_rel(source: Path, target: Path, base_dir: Path, dryrun: bool = False):
    """if both source/target under base_dir, create link with relative path.

    why it's tricky for that?

    source maybe file or directory. maybe relative or absolute.
    target maybe file or directory. maybe relative or absolute.
    base_dir maybe relative or absolute.
    """
    # should relative
    base_dir = base_dir.resolve()
    target.parent.mkdir(exist_ok=True, parents=True)
    if source.resolve().is_relative_to(base_dir) and target.resolve().is_relative_to(base_dir):
        source = source.absolute()
        target = target.absolute()
        source = Path(os.path.relpath(source, target.parent))

    if target.is_symlink() or target.exists():
        target.unlink()
    if not dryrun:
        target.symlink_to(source, target_is_directory=source.is_dir())


def softlink_apply_patches():
    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path of the patches/triton directory
    patches_triton_dir = os.path.join(script_dir, '..', 'patches', 'triton')
    # Construct the path of the 3rdparty/triton directory
    _3rdparty_triton_dir = os.path.join(script_dir, '..', '3rdparty', 'triton')

    # Check if the patches/triton directory exists
    if not os.path.exists(patches_triton_dir):
        print(f"Patches directory {patches_triton_dir} does not exist.")
        return

    # Traverse all files and folders in the patches/triton directory
    for root, dirs, files in os.walk(patches_triton_dir):
        # Calculate the relative path
        relative_path = os.path.relpath(root, patches_triton_dir)
        # Construct the path of the target directory
        target_dir = os.path.join(_3rdparty_triton_dir, relative_path)

        # Create the target directory if it does not exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Copy files
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)
            # Check if the source file is a relative path
            base_dir = Path(__file__).parent.parent
            create_symlink_rel(Path(source_file), Path(target_file), base_dir)
