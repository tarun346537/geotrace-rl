# Dependency profiles

`base.txt` installs the project from the repository root and delegates base
dependency selection to package metadata. That metadata selects NumPy 1.26.4 on
Python 3.11 and 2.2.6 on Python 3.12 or newer. Run
`python -m pip install -r requirements/base.txt` from the project root.
`cpu.txt` records the direct
package versions observed in the local Python 3.12 checks. `dev.txt` holds the
linter, type checker and test tools. `constraints-cpu.txt` pins their transitive dependencies for the CPU/test environment. Package metadata uses the same exact version pins:
Python 3.11 selects the legacy training versions, while Python 3.12 or newer
selects the current CPU-compatible versions. Local checks used Python 3.12.

`cuda-verl.txt` is a separate Linux x86_64 / Python 3.11 / CUDA 12.4 profile for
the pinned verl trainer. Do not merge it with the CPU profile: verl 0.5.0 requires
NumPy below 2, and vLLM 0.8.5 requires PyTorch 2.6.0. Its Ray constraint excludes
2.44.x. The selected Ray 2.45.0 satisfies both packages. Transformers 4.51.3 and
tokenizers 0.21.1 satisfy that vLLM release's minimum versions.

Both profiles use SimPEG 0.24.0 and discretize 0.11.3. The CPU profile uses
NumPy 2.2.6 and PyTorch 2.8.0, while the CUDA profile retains the versions
required by verl and vLLM. Run the installation and `pip check` commands on
each cluster node.

`cuda-kernels.txt` installs FlashAttention after PyTorch is present. Its source
build needs the CUDA toolkit and matching compiler/ABI. Use
`--no-build-isolation` for this separate step. Do not let a kernel installation
silently replace the pinned PyTorch wheel.

Primary compatibility references:

- [verl 0.5.0 package requirements](https://github.com/verl-project/verl/blob/v0.5.0/setup.py)
- [vLLM 0.8.5 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.8.5/requirements/cuda.txt)
- [vLLM 0.8.5 common requirements](https://github.com/vllm-project/vllm/blob/v0.8.5/requirements/common.txt)
- [SimPEG 0.24.0](https://pypi.org/project/simpeg/0.24.0/)
- [discretize 0.11.3](https://pypi.org/project/discretize/0.11.3/)
