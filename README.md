# OmniBioAI Dev Environment

> Full AI/Bioinformatics development environment for GPU/DGX machines.
> Built on NVIDIA PyTorch 25.10 with CUDA support.

[![Docker](https://img.shields.io/badge/ghcr.io-omnibioai--dev--env-blue?logo=docker)](https://ghcr.io/man4ish/omnibioai-dev-env)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GPU Required](https://img.shields.io/badge/GPU-required-orange?logo=nvidia)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## What This Repository Is

This is an **optional developer convenience environment** — think of it as:

> **"A researcher's laptop in a container."**

It is intended for:
- OmniBioAI contributors and researchers
- Exploratory AI/LLM prototyping
- Single-cell and multi-omics experimentation
- Notebook-driven development (Python + R)
- GPU-accelerated model training and inference

## What This Repository Is NOT

This is **not** part of the OmniBioAI production stack. It is not:
- A Tool Execution Service (TES) runtime
- A production or cloud deployment image
- Used by OmniBioAI pipelines at runtime

| Component | Role |
|---|---|
| OmniBioAI Workbench | Production platform |
| Tool Execution Service (TES) | Stateless execution |
| Tool Runtime Images | Minimal, contract-driven |
| **This Repository** | Interactive development only |

---

## What's Inside

| Category | Tools |
|---|---|
| **Deep Learning** | PyTorch 2.9 (GPU), TorchVision, TensorRT, Flash Attention, Transformer Engine |
| **ML/Data Science** | Scikit-learn, XGBoost, LightGBM, Polars, Pandas, NumPy, SciPy |
| **Bioinformatics** | GATK 4.5, Samtools, BCFTools, FastQC, SnpEff, Nextflow, BEDTools |
| **Genomics/R** | R 4.x, Bioconductor, DESeq2, limma, edgeR, ComplexHeatmap, scran, scater |
| **LLM/AI** | Transformers, HuggingFace Hub, Accelerate, Safetensors, Ollama |
| **Visualization** | Matplotlib, Seaborn, Plotly, Bokeh, TensorBoard |
| **Notebook** | JupyterLab 4.x (pre-configured, GPU-enabled) |

---

## Requirements

- NVIDIA GPU (A100, H100, or DGX system recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on host
- Docker with GPU support
- 50GB+ free disk space

---

## Quick Start

### Option A — Pull from GHCR (recommended)
```bash
docker pull ghcr.io/man4ish/omnibioai-dev-env:latest
bash run_ai_dev.sh
```

### Option B — Build locally
```bash
git clone https://github.com/man4ish/omnibioai-dev-docker
cd omnibioai-dev-docker
docker build -t ghcr.io/man4ish/omnibioai-dev-env:latest .
bash run_ai_dev.sh
```

---

## Running the Container

```bash
bash run_ai_dev.sh [OPTIONS]

Options:
  --jupyter    Start JupyterLab automatically
  --ollama     Start Ollama server automatically
  --build      Force rebuild image from Dockerfile
  --help       Show help
```

This launches an interactive container with:
- Full GPU access (`--gpus all`)
- Shared memory for PyTorch DataLoader (`--ipc=host`)
- HuggingFace cache mounted (`~/.cache/huggingface`)
- Ollama models mounted (`~/.ollama`)
- Current directory mounted as `/workspace`
- JupyterLab on port `8888`
- Ollama server on port `11434`

### Start JupyterLab:
```bash
# Via flag (recommended)
bash run_ai_dev.sh --jupyter

# Or manually inside container
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Then open: `http://localhost:8888`

### Start Ollama:
```bash
# Via flag
bash run_ai_dev.sh --ollama

# Or manually inside container
ollama serve &
ollama pull llama3
ollama run llama3 "Summarize the role of TP53 in cancer"
```

---

## GPU Validation

Inside the container, verify CUDA and all tools:

```bash
# PyTorch GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Bioinformatics tools
nextflow -version
gatk --version
samtools --version | head -1
fastqc --version
snpeff -version

# R + Bioconductor
R -e "library(DESeq2); packageVersion('DESeq2')"
```

If CUDA returns `False`, verify:
- NVIDIA drivers are installed
- Docker is configured with GPU support
- `nvidia-container-toolkit` is available

---

## HuggingFace Authentication

```bash
# Inside container
huggingface-cli login

# Or via environment variable
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

---

## Example — GPU-accelerated single-cell analysis

```python
import torch
import scanpy as sc

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

adata = sc.read_h5ad("/workspace/data/sample.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata)
```

---

## Typical Use Cases

- Prototyping new OmniBioAI plugins
- Exploratory single-cell analysis (Scanpy/Seurat workflows)
- LLM prompt and RAG experimentation
- Model benchmarking and training
- Notebook-based research
- Rapid testing before productionization

---

## Repository Structure

| File | Description |
|---|---|
| `Dockerfile` | Main image definition |
| `requirements.txt` | Portable Python dependencies |
| `requirements.dgx.txt` | DGX-specific packages (pre-installed in base, docs only) |
| `run_ai_dev.sh` | Container launch script |
| `tests/` | Validation tests |
| `pyproject.toml` | Project metadata |
| `.gitignore` | Prevents secrets/caches from being committed |
| `.dockerignore` | Prevents data/models from being baked into image |

---

## Part of the OmniBioAI Ecosystem

This dev environment is designed to work alongside the
[OmniBioAI platform](https://github.com/man4ish/omnibioai) —
a unified AI-powered bioinformatics workbench supporting:

- 97 bioinformatics plugins
- RNA-seq, single-cell, spatial omics, variant calling
- TES workflow execution (Slurm, K8s, AWS Batch, Azure)
- RAG-powered literature search (PubMed + FAISS)
- ML model registry
- Multi-cloud support (AWS, Azure, GCP)

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)

---

## Citation

If you use this environment in your research, please cite:

```
OmniBioAI Dev Environment (2025)
Manish Kumar
https://github.com/man4ish/omnibioai-dev-docker
```
