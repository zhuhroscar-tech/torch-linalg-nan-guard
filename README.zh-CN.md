# torch-linalg-nan-guard

本仓库已合并到 [`zhuhroscar-tech/torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards)。

请改用统一的 umbrella package：

```bash
git clone https://github.com/zhuhroscar-tech/torch-correctness-guards.git
cd torch-correctness-guards
python3 -m pip install -e ".[torch]"
torch-guard run linalg-nan
```

Python 导入也迁移到了合并后的包：

```python
from torch_correctness_guards import safe_svdvals, safe_eigvalsh, safe_vector_norm, safe_norm
```

原 `torch_linalg_nan_guard` 实现已迁移为 `torch_correctness_guards.guards.linalg_nan`，测试与 CLI 覆盖均已保留。本仓库将归档，以便公开账号聚焦维护统一的高质量包。

[MIT 许可证](LICENSE)。
