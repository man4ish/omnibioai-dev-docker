# OmniBioAI Developer Environment

**GPU-Enabled AI & Bioinformatics Development Docker Image**

This repository provides a **fully self-contained, GPU-enabled development environment** for AI, bioinformatics, and data science workflows used during **OmniBioAI development and research**.

It is designed to replicate a **powerful research workstation in a single Docker image**, enabling rapid experimentation with machine learning, LLMs, single-cell analysis, and exploratory bioinformatics — without polluting the host system.

---

## What This Repository Is

This repository is an **optional developer convenience environment**.

It is intended for:

* OmniBioAI contributors
* Researchers doing exploratory analysis
* AI / LLM prototyping
* Single-cell and multi-omics experimentation
* Notebook-driven development (Python + R)
* GPU-accelerated model training and inference

Think of it as:

> **“A researcher’s laptop in a container.”**

---

## What This Repository Is NOT

This repository is **not** part of the OmniBioAI production stack.

Specifically, it is **not**:

* A Tool Execution Service (TES) runtime
* A production container image
* A workflow execution environment
* A cloud deployment image
* Used by OmniBioAI pipelines or agents at runtime

Execution containers used by TES are **minimal, stateless, and contract-driven**
This image is **stateful, interactive, and intentionally heavy**

---

## Environment Overview

This image is built on **NVIDIA PyTorch 25.10** with full CUDA support and includes a carefully curated stack for AI and computational biology.

### Core Capabilities

* GPU-accelerated Python development
* R-based statistical and single-cell analysis
* Interactive JupyterLab environment
* Local database support for prototyping
* LLM tooling for local and hybrid inference

---

## Included Stack

### Base

* NVIDIA PyTorch 25.10 (CUDA-enabled)
* Ubuntu base image
* CUDA + cuDNN preconfigured

### Languages

* Python
* R

### Machine Learning & Data Science

* PyTorch
* Scikit-learn
* XGBoost
* LightGBM
* Polars
* NumPy / Pandas

### Visualization

* Matplotlib
* Seaborn
* Plotly
* Bokeh

### LLM / AI Tooling

* Hugging Face Transformers
* Hugging Face Hub
* Accelerate
* Safetensors
* Ollama (client-side integration)

### Databases

* MySQL (local development and prototyping)

### Interactive Development

* JupyterLab (preconfigured)

---

## Repository Structure

```
.
├── Dockerfile          # Defines the full AI development environment
├── requirements.txt    # Python dependencies
├── run_ai_dev.sh       # Convenience script to run the container
├── README.md
└── .gitignore
```

---

## Build the Docker Image

```bash
docker build -t omnibioai-dev-env .
```

You may choose a different tag if desired.

---

## Run the Development Container

Use the provided helper script:

```bash
bash run_ai_dev.sh
```

### What the Run Script Does

The script mounts:

* Hugging Face cache and authentication
* Ollama model directory
* Current working directory as the workspace

This enables:

* Persistent model downloads
* Token reuse
* Seamless local development

---

## Ollama Integration

The container is configured to **connect to an existing Ollama instance** by default.

The Ollama port (`11434`) is commented out in the run script.

If you want the container itself to host Ollama:

1. Uncomment the port mapping
2. Start Ollama inside the container

---

## Hugging Face Authentication

Authenticate once using:

```bash
huggingface-cli login
```

Or set the token via environment variable:

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

The token will be mounted into the container securely.

---

## GPU Validation

After starting the container, verify CUDA support:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

If this prints `False`, verify:

* NVIDIA drivers are installed
* Docker is configured with GPU support
* `nvidia-container-toolkit` is available

---

## Typical Use Cases

This environment is ideal for:

* Prototyping new OmniBioAI plugins
* Exploratory single-cell analysis (Scanpy / Seurat workflows)
* LLM prompt and RAG experimentation
* Model benchmarking
* Notebook-based research
* Rapid testing before productionization

---

## Relationship to OmniBioAI Platform

| Component                    | Role                         |
| ---------------------------- | ---------------------------- |
| OmniBioAI Workbench          | Production platform          |
| Tool Execution Service (TES) | Stateless execution          |
| Tool Runtime Images          | Minimal, contract-driven     |
| **This Repository**          | Interactive development only |

This separation is **intentional** and critical for reproducibility.

---

## License & Usage

This repository is provided as a **development convenience**.

You are free to:

* Modify it
* Extend it
* Use it as a base for your own research images

Just remember:
**Do not use this image in production pipelines.**

---

## Final Note

As OmniBioAI grows, reproducibility and architectural clarity matter more than convenience.

This repository exists to **accelerate development**, not to blur system boundaries.

That distinction is deliberate.

