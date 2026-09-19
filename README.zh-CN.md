# torch-linalg-nan-guard

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

两个针对 PyTorch `torch.linalg` 静默正确性缺陷的独立防护：

1. 在调用 `torch.linalg.svdvals()` 或 `torch.linalg.eigvalsh()` 前拒绝包含 NaN、Inf 的输入——这两个函数可能对无效输入静默返回看似正常的有限值，而不是报错或传播 NaN。
2. 在不触发 PyTorch CPU float32 累加器缺陷的前提下计算 `torch.linalg.vector_norm()` / `torch.norm()`（L2/Frobenius 范数）——该缺陷会让普通的 float16/bfloat16/float32 输入静默丢失精度，或溢出为 `inf`（并导致反向传播时梯度静默归零）。

两个防护都是显式调用的包装函数，配合一个 CLI 检查当前安装的 PyTorch 实际行为；均不会全局修改 PyTorch。

## 安装与快速上手

需要 Python 3.9+ 和兼容的 PyTorch 2.0+；实际支持的 Python 版本也取决于所安装的 PyTorch wheel。

```bash
git clone https://github.com/zhuhroscar-tech/torch-linalg-nan-guard.git
cd torch-linalg-nan-guard
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[torch]"
torch-linalg-nan-guard --json
```

如果已有针对特定平台配置的 PyTorch 环境，可直接在该环境执行 `pip install -e .`。

```python
import torch
from torch_linalg_nan_guard import safe_svdvals, safe_eigvalsh, safe_vector_norm, safe_norm

matrix = torch.eye(3, dtype=torch.float64)
singular_values = safe_svdvals(matrix)
eigenvalues = safe_eigvalsh(matrix, UPLO="L")

x = torch.randn(50_000_000, dtype=torch.float32)
norm = safe_vector_norm(x, ord=2)   # 或 safe_norm(x)
```

输入为有限值时，`safe_svdvals`/`safe_eigvalsh` 直接调用 PyTorch 原函数。`safe_vector_norm`/`safe_norm` 对 `float64`/复数输入以及非 L2 的 `ord`/`p` 也是直接透传；对受影响的 dtype 做 L2 范数计算时，内部会用 `float64` 累加平方和后再转换回原 dtype。形状、dtype、设备及对称性等要求仍以 PyTorch 为准。

## 如何理解诊断结果

不带参数运行 `torch-linalg-nan-guard` 会同时执行两项诊断。`--skip-norm-check` 可跳过范数精度/溢出诊断（默认会对 1600 万元素做一次 CPU 归约，在资源受限的主机上会占用真实的时间和内存）。

- **NaN 静默吞没诊断**：在小型 CPU 矩阵上对比“仅计算数值”的操作与“完整分解”操作。JSON 会区分“原操作抛出异常”和“静默返回有限值”，并记录防护函数是否报错。
- **范数精度诊断**：在若干元素规模下对比 `torch.linalg.vector_norm()`，以及一个刻意构造的大数值量级用例，与基于同一输入计算的独立 `float64` 参考值对比，报告每个规模下防护前后的相对误差，以及是否溢出为 `inf`。

结果取决于所安装的 PyTorch 版本/后端；某次检查通过并不代表所有输入、dtype 或设备都安全。两项诊断均已在本项目自己的 macOS CPU 主机上从零复现（具体复现方式见测试代码），并在 `ubuntu-latest` CI 上验证；两项诊断都不覆盖 GPU/MPS 特有的累加器行为。

相关报告：
- NaN 静默吞没：[PyTorch #187759](https://github.com/pytorch/pytorch/issues/187759)、[NumPy #20280](https://github.com/numpy/numpy/issues/20280)。
- 范数精度/溢出：[PyTorch #169237](https://github.com/pytorch/pytorch/issues/169237)（精度损失；对应的修复提案 [PR #169996](https://github.com/pytorch/pytorch/pull/169996) 在评审后被关闭且未合并），[PyTorch #193006](https://github.com/pytorch/pytorch/issues/193006)（溢出/梯度静默归零；对应的修复提案 [PR #194326](https://github.com/pytorch/pytorch/pull/194326) 仍是未合并的开放草案）。截至目前两个 issue 均处于打开状态。

## 限制与开发

- 仅保护 `svdvals`/`eigvalsh`（NaN 静默吞没）以及 `vector_norm`/`norm` 的 L2/Frobenius 用法（累加器精度/溢出），不是通用的数值正确性检查器。
- NaN 防护：检查范围是整个 tensor；batch 内任意元素无效都会拒绝整次调用，不定位具体矩阵。
- 范数防护：内部的 `float64` 中间缓冲区相比原生（有缺陷的）路径会让归约的峰值内存大致翻倍，且其本身在极端量级下仍受 `float64` 自身（远大得多，但并非无限）精度和范围的限制。
- 扫描/类型转换有额外开销，`.item()` 也可能触发加速器同步；两项诊断都只在 CPU 上针对 float16/bfloat16/float32 张量复现和防护，均不覆盖 GPU 特有行为。

```bash
pip install -e ".[dev,torch]"
pytest -v --cov=torch_linalg_nan_guard --cov-report=term-missing
```

[MIT 许可证](LICENSE)。
