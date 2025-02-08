import functools
import os
import pathlib
import re
import subprocess
import sysconfig


NVSHMEM_WRAPPER_FUNCS = [
    "nvshmem_ptr_wrapper",
    "nvshmem_my_pe_wrapper",
    "nvshmem_n_pes_wrapper",
    "nvshmem_int_p_wrapper",
]

NVSHMEM_WRAPPER_EXTERN = {
    "nvshmem_ptr_wrapper": """.extern .func (.param .b64 func_retval0) nvshmem_ptr_wrapper
(
    .param .b64 nvshmem_ptr_wrapper_param_0,
    .param .b32 nvshmem_ptr_wrapper_param_1
);""",
    "nvshmem_my_pe_wrapper": """.extern.func (.param.b32 func_retval0) nvshmem_my_pe_wrapper();""",
    "nvshmem_n_pes_wrapper": """.extern.func (.param.b32 func_retval0) nvshmem_n_pes_wrapper();""",
    "nvshmem_int_p_wrapper": """.extern.func nvshmem_int_p_wrapper
(
   .param.b64 nvshmem_int_p_wrapper_param_0,
   .param.b32 nvshmem_int_p_wrapper_param_1,
   .param.b32 nvshmem_int_p_wrapper_param_2
);""",
}


@functools.lru_cache()
def _path_to_binary(binary: str):
    binary += sysconfig.get_config_var("EXE")
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
    ]

    paths += [os.environ.get("CUDA_HOME", "/usr/local/cuda/") + "bin/nvlink"]

    for path in paths:
        if os.path.exists(path) and os.path.isfile(path):
            result = subprocess.check_output(
                [path, "--version"], stderr=subprocess.STDOUT
            )
            if result is not None:
                version = re.search(
                    r".*release (\d+\.\d+).*",
                    result.decode("utf-8"),
                    flags=re.MULTILINE,
                )
                if version is not None:
                    return path, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_nvshmem_home():
    return pathlib.Path(
        os.environ.get(
            "NVSHMEM_HOME",
            pathlib.Path(__file__).parent.parent.parent.parent
            / "nvshmem"
            / "build"
            / "install",
        )
    )


@functools.lru_cache()
def get_nvshmem_lib():
    return get_nvshmem_home() / "lib"


@functools.lru_cache()
def get_triton_nvshmem_runtime_lib():
    pass


@functools.lru_cache()
def get_nvshmem_cubin(capability):
    return (
        pathlib.Path(__file__).parent.parent.parent
        / "runtime"
        / f"nvshmem_wrapper.sm{capability}.cubin"
    )


@functools.lru_cache()
def get_ptxas(arch: int):
    name = "ptxas-blackwell" if arch >= 100 else "ptxas"
    return _path_to_binary(name)


@functools.lru_cache()
def get_nvlink(arch: int):
    # name = "ptxas-blackwell" if arch >= 100 else "nvlink"
    return _path_to_binary("nvlink")


def has_nvshmem_wrappers(ptx):
    return any([x in ptx for x in NVSHMEM_WRAPPER_FUNCS])


def get_nvshmem_wrappers(ptx):
    funcs = []
    for func in NVSHMEM_WRAPPER_FUNCS:
        if func in ptx:
            funcs.append(func)
    return funcs


def patch_nvshmem_wrapper_externs(ptx):
    wrappers = get_nvshmem_wrappers(ptx)
    if len(wrappers) == 0:
        return ptx
    externs = []
    for wrapper in wrappers:
        externs.append(NVSHMEM_WRAPPER_EXTERN[wrapper])
    externs = "\n".join(externs)

    MARKER = ".address_size 64"
    loc = [x.strip() for x in ptx.split("\n")].index(MARKER) + 1
    assert loc > 0

    lines = ptx.split("\n")
    ptx = "\n".join(lines[:loc]) + "\n" + externs + "\n" + "\n".join(lines[loc:])
    return ptx

