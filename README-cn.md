# Triton-distributed

[原始Triton README](upstream-README.md) | [英文README](README.md)

Triton-distributed是基于OpenAI Triton构建的分布式编译器，专为计算-通信重叠优化设计。

使用Triton-distributed，开发者可以创建性能媲美优化库（如NVIDIA的[Distributed-GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm)和字节跳动的[FLUX](https://github.com/bytedance/flux/blob/main/README.md)）的高效Kernel。当前主要支持NVIDIA GPU和AMD GPU，也可移植到其他硬件平台。如需在自定义硬件上使用，请联系我们。

## 快速入门
### 源码安装
#### 推荐环境：
- Python 3.9（建议使用虚拟环境）
- CUDA 12.4
- PyTorch 2.4

*注：其他版本依赖可能兼容，但不做保证。如遇安装问题请在GitHub Issues反馈*

#### 安装步骤：
1. 克隆仓库：
   ```sh
   git clone https://github.com/your-org/Triton-distributed /path/to/Triton-distributed
   ```
2. 初始化子模块：
   ```sh
   cd /path/to/Triton-distributed
   git submodule update --init --recursive
   ```
3. 安装依赖：
   ```sh
   pip3 install torch==2.4 ninja cmake wheel pybind11 cuda-python==12.4 numpy
   ```
4. 构建安装：
   ```sh
   pip3 install -e python --verbose --no-build-isolation
   ```
5. 配置环境（每次使用前执行）：
   ```sh
   source scripts/setenv.sh
   ```

### 验证安装
#### 单节点AllGather GEMM
需要8块H800 GPU：
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/test_ag_gemm_intra_node.py --case correctness_tma
```

#### 单节点ReduceScatter GEMM
需要8块H800 GPU：
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/test_gemm_rs_intra_node.py 8192 8192 29568 --check
```

#### NVSHMEM功能验证：
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/test_nvshmem_api.py
```

## 使用指南
Triton-distributed提供两个抽象层用于开发计算-通信重叠的分布式Kernel：
- **底层原语**（已开放）
- **高层原语**（即将开发）

请见`triton.distributed.language`。此外，所有NVSHMEM设备侧操作通过`triton.language.extra.libshmem_device`暴露：

### 底层原语
#### 上下文管理
```python
rank(axis=-1, _builder=None)
num_ranks(axis=-1, _builder=None)
symm_at(ptr, rank, _builder=None)
```

#### 同步控制
```python
wait(barrierPtrs, numBarriers, scope: str, semantic: str, _builder=None)
consume_token(value, token, _builder=None)
notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _builder=None)
```

#### NVSHMEM原语

```python
my_pe()
n_pes()
int_p(dest, value, pe)
remote_ptr(local_ptr, pe)
barrier_all()
barrier_all_block()
barrier_all_warp()
sync_all()
sync_all_block()
sync_all_warp()
quiet()
fence()
getmem_nbi_block(dest, source, bytes, pe)
getmem_block(dest, source, bytes, pe)
getmem_nbi_warp(dest, source, bytes, pe)
getmem_warp(dest, source, bytes, pe)
getmem_nbi(dest, source, bytes, pe)
getmem(dest, source, bytes, pe)
putmem_block(dest, source, bytes, pe)
putmem_nbi_block(dest, source, bytes, pe)
putmem_warp(dest, source, bytes, pe)
putmem_nbi_warp(dest, source, bytes, pe)
putmem(dest, source, bytes, pe)
putmem_nbi(dest, source, bytes, pe)
putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
signal_op(sig_addr, signal, sig_op, pe)
signal_wait_until(sig_addr, cmp_, cmp_val)
```

*示例：环形通信*
```python
@triton.jit
def ring_put(ptr):
    mype = libshmem_device.my_pe()
    npes = libshmem_device.n_pes()
    peer = (mype + 1) % npes
    libshmem_device.int_p(ptr, mype, peer)
```

### 高层原语（即将发布）
基于分块设计理念的高层抽象（详见[MLSys 2025论文](https://mlsys.org/virtual/2025/poster/2969)）将在会议后发布，进一步简化分布式Kernel开发。

## Roadmap
### 功能
- [x] Release low-level primitives
- [ ] Release high-level primitives
### Kernels
- [x] Release single-node GEMM TP overlapping kernels
- [ ] Release single-node MoE TP overlapping kernels
- [ ] Release single-node distributed Flash-Decoding kernels
- [ ] Release single-node MoE EP overlapping kernels
- [ ] Release cross-node GEMM TP overlapping kernels
- [ ] Release cross-node MoE TP overlapping kernels
- [ ] Release cross-node distributed Flash-Decoding kernels
- [ ] Release cross-node EP all-to-all kernels (similar to [DeepEP](https://github.com/deepseek-ai/DeepEP))
### 后端
- [x] Nvidia SM90a support
- [x] Nvidia SM80 support
- [ ] Nvidia SM89 support
- [ ] AMD support
### 性能
- [ ] Performance report

## 引用
如在学术研究中使用Triton-distributed，请引用：
```bibtex
@misc{zheng2025tilelink,
      title={TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives}, 
      author={Size Zheng and Jin Fang and Xuegui Zheng and Qi Hou and Wenlei Bao and Ningxin Zheng and Ziheng Jiang and Dongyang Wang and Jianxi Ye and Haibin Lin and Li-Wen Chang and Xin Liu},
      year={2025},
      eprint={TBD},
      archivePrefix={MLSys}
}
```

## 许可协议
MIT License

---
