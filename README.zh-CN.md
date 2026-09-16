# torch-linalg-nan-guard

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

在调用 PyTorch 的 `torch.linalg.svdvals()` 或 `torch.linalg.eigvalsh()` 前，检查并拒绝包含 NaN、Inf 的输入。项目提供两个显式调用的包装函数，以及用于检查当前 PyTorch 安装行为的 CLI。

部分数值计算后端可能对无效输入返回看似正常的有限值，另一些后端则会报错。本工具统一采用明确规则：分解前发现非有限值就抛出 `ValueError`，不会全局修改 PyTorch。

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
from torch_linalg_nan_guard import safe_svdvals, safe_eigvalsh

matrix = torch.eye(3, dtype=torch.float64)
singular_values = safe_svdvals(matrix)
eigenvalues = safe_eigvalsh(matrix, UPLO="L")
```

输入全部为有限值时，包装函数直接调用 PyTorch。`safe_svdvals` 支持 `driver`，`safe_eigvalsh` 支持 `UPLO`；形状、dtype、设备及对称性等要求仍以 PyTorch 为准。

## 如何理解诊断结果

CLI 使用小型 CPU 矩阵，对比仅计算数值的操作和完整分解操作。JSON 会区分“原操作抛出异常”和“静默返回有限值”，并记录防护函数是否报错。结果取决于本机后端，某个测试通过并不代表所有输入或设备都安全。

相关报告：[PyTorch #187759](https://github.com/pytorch/pytorch/issues/187759)、[NumPy #20280](https://github.com/numpy/numpy/issues/20280)。

## 限制与开发

- 仅保护上述两个操作，不是通用的数值正确性检查器。
- 检查范围是整个 tensor；batch 内任意元素无效都会拒绝整次调用，不定位具体矩阵。
- 扫描有额外开销，`.item()` 也可能触发加速器同步；CPU 诊断不覆盖 GPU 特有行为。

```bash
pip install -e ".[dev,torch]"
pytest -v --cov=torch_linalg_nan_guard --cov-report=term-missing
```

[MIT 许可证](LICENSE)。
