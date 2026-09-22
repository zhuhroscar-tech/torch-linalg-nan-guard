# torch-linalg-nan-guard

This repository has been consolidated into [`zhuhroscar-tech/torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards).

Use the umbrella package instead:

```bash
git clone https://github.com/zhuhroscar-tech/torch-correctness-guards.git
cd torch-correctness-guards
python3 -m pip install -e ".[torch]"
torch-guard run linalg-nan
```

Python imports moved to the consolidated package:

```python
from torch_correctness_guards import safe_svdvals, safe_eigvalsh, safe_vector_norm, safe_norm
```

The original `torch_linalg_nan_guard` implementation has been migrated as `torch_correctness_guards.guards.linalg_nan`, with tests and CLI coverage preserved. This repository is archived to keep the public account focused on maintained umbrella packages.

[MIT license](LICENSE).
