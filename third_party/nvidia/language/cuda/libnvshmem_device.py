from triton.language import core


@core.extern
def my_pe(_builder=None):
    return core.extern_elementwise("libnvshmem_device", "", [], {
        (): ("nvshmem_my_pe", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def n_pes(_builder=None):
    return core.extern_elementwise("libnvshmem_device", "", [], {
        (): ("nvshmem_n_pes", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int_p(dest, value, pe, _builder=None):
    # force have a return value, even not used.
    return core.extern_elementwise(
        "libnvshmem_device", "", [dest, value, pe], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32"), core.dtype("int32")):
            ("nvshmem_int_p", core.dtype("int32")),
        }, is_pure=False, _builder=_builder, check_args=False)


@core.extern
def remote_ptr(local_ptr, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device", "", [local_ptr, pe], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32")):
            ("nvshmem_ptr", core.pointer_type(core.dtype("int32"))),
        }, is_pure=False, _builder=_builder, check_args=False)
