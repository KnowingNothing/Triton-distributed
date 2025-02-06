import triton
import triton.language as tl
from triton.language import core


def patch_triton_module(func):
    func.__module__ = f"triton.{func.__module__}"
    return func


@patch_triton_module
@core.extern
def nvshmem_ptr(ptr, peer, _builder=None):
    return core.inline_asm_elementwise(
        asm=f"""
        {{
          .reg .b32 temp_param_reg;
          .param .b64 param0;
          st.param.b64 	[param0+0], $1;
          .param .b32 param1;
          st.param.b32 	[param1+0], $2;
          .param .b64 retval0;
          call.uni (retval0),
          nvshmem_ptr_wrapper,
          (
          param0,
          param1
          );
          ld.param.b64 	$0, [retval0+0];
        }}
        """,
        constraints=("=l,l,r"),  # force have a return value, even not used.
        args=[ptr, peer],
        dtype=tl.uint64,
        is_pure=True,  # no optimize this!
        pack=1,
        _builder=_builder,
    )


@patch_triton_module
@core.extern
def nvshmem_my_pe_wrapper(_builder=None):
    return core.inline_asm_elementwise(
        asm=f"""
        {{
            .reg .b32 temp_param_reg;
            .param .b32 retval0;
            call.uni (retval0),
            nvshmem_my_pe_wrapper,
            (
            );
            ld.param.b32 	$0, [retval0+0];
        }}
        """,
        constraints=("=r"),  # force have a return value, even not used.
        args=[],
        dtype=tl.uint32,
        is_pure=True,  # no optimize this!
        pack=1,
        _builder=_builder,
    )


@patch_triton_module
@core.extern
def nvshmem_n_pes_wrapper(_builder=None):
    return core.inline_asm_elementwise(
        asm=f"""
        {{
            .reg .b32 temp_param_reg;
            .param .b32 retval0;
            call.uni (retval0),
            nvshmem_n_pes_wrapper,
            (
            );
            ld.param.b32 	$0, [retval0+0];
        }}
        """,
        constraints=("=r"),  # force have a return value, even not used.
        args=[],
        dtype=tl.uint32,
        is_pure=True,  # no optimize this!
        pack=1,
        _builder=_builder,
    )


@patch_triton_module
@core.extern
def nvshmem_int_p_wrapper(ptr, mype, peer, _builder=None):
    return core.inline_asm_elementwise(
        asm=f"""
        {{
            .reg .b32 temp_param_reg;
            .param .b64 param0;
            st.param.b64 	[param0+0], $1;
            .param .b32 param1;
            st.param.b32 	[param1+0], $2;
            .param .b32 param2;
            st.param.b32 	[param2+0], $3;
            call.uni
            nvshmem_int_p_wrapper,
            (
            param0,
            param1,
            param2
            );
            mov.b32 $0, 0;
        }}
        """,
        constraints=("=r,l,r,r"),  # force have a return value, even not used.
        args=[ptr, mype, peer],
        dtype=tl.uint32,
        is_pure=False,  # no optimize this!
        pack=1,
        _builder=_builder,
    )
