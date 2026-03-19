# ==========================================================
# Image Name: ai-dev
# Description: Full AI Development Environment
# Base: NVIDIA PyTorch 25.10 (CUDA-enabled)
# Includes: Python (Data Science Stack) + R (Bioconductor) + MySQL + Jupyter + Ollama
# Maintainer: Manish Kumar
# ==========================================================

FROM nvcr.io/nvidia/pytorch:25.10-py3

# DL3048 fix: use lowercase dotted label keys (OCI spec)
LABEL org.opencontainers.image.authors="Manish Kumar"
LABEL org.opencontainers.image.description="AI-Dev: PyTorch 25.10 base with R, MySQL, Jupyter, and Ollama for full-stack AI and bioinformatics development."
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.created="2025-11-08"

WORKDIR /workspace

# ==========================================================
# 1. System dependencies
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libmysqlclient-dev \
    mysql-server \
    software-properties-common \
    dirmngr \
    gnupg \
    apt-transport-https \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ==========================================================
# 2. R Installation + Bioconductor packages
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends r-base \
 && rm -rf /var/lib/apt/lists/* \
 && R -e "install.packages(c('tidyverse', 'data.table', 'BiocManager'), repos='https://cloud.r-project.org')" \
 && R -e "BiocManager::install(c('ComplexHeatmap', 'limma', 'edgeR', 'DESeq2'), ask=FALSE)"

# ==========================================================
# 3. Python Data Science Stack
# DL3013 fix: use requirements file for pinned versions
# DL3042 fix: --no-cache-dir on all pip calls
# ==========================================================
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ==========================================================
# 4. MySQL Configuration
# DL3001 fix: removed `service mysql stop` — meaningless in a container
#             (no init system; MySQL never starts during build)
# ==========================================================
RUN mkdir -p /var/lib/mysql /var/run/mysqld \
 && chown -R mysql:mysql /var/lib/mysql /var/run/mysqld \
 && chmod 777 /var/run/mysqld

# ==========================================================
# 5. Ollama Installation
# DL4001 fix: use curl consistently (wget removed from this layer)
# ==========================================================
RUN curl -fsSL https://ollama.com/install.sh | bash \
 && ollama --version

# ==========================================================
# 6. Port Configuration
# 3306 -> MySQL  |  8888 -> JupyterLab  |  11434 -> Ollama
# ==========================================================
EXPOSE 3306
EXPOSE 8888
EXPOSE 11434

# ==========================================================
# 7. Jupyter Configuration
# ==========================================================
RUN mkdir -p /root/.jupyter \
 && printf '%s\n' \
    "c.NotebookApp.ip = '0.0.0.0'" \
    "c.NotebookApp.open_browser = False" \
    "c.NotebookApp.allow_root = True" \
    >> /root/.jupyter/jupyter_notebook_config.py

# ==========================================================
# 8. Bioinformatics tools — system deps
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    default-jdk \
    fastqc \
    samtools \
    bcftools \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# ==========================================================
# 9. Nextflow
# DL4001 fix: curl only (removed wget from this layer)
# ==========================================================
RUN curl -fsSL https://get.nextflow.io | bash \
 && mv nextflow /usr/local/bin/ \
 && chmod +x /usr/local/bin/nextflow

# ==========================================================
# 10. GATK 4.5.0.0
# DL4001 fix: wget only (removed curl from this layer)
# ==========================================================
RUN wget -q https://github.com/broadinstitute/gatk/releases/download/4.5.0.0/gatk-4.5.0.0.zip \
 && unzip gatk-4.5.0.0.zip -d /opt \
 && ln -s /opt/gatk-4.5.0.0/gatk /usr/local/bin/gatk \
 && rm gatk-4.5.0.0.zip

# ==========================================================
# 11. SnpEff 5.3a
# DL4001 fix: wget only (consistent with GATK layer above)
# ==========================================================
RUN wget -q -O /tmp/snpEff_latest_core.zip \
    https://snpeff.odsp.astrazeneca.com/versions/snpEff_latest_core.zip \
 && unzip /tmp/snpEff_latest_core.zip -d /opt \
 && ln -s /opt/snpEff/snpEff.jar /usr/local/bin/snpEff.jar \
 && printf '#!/bin/bash\njava -jar /usr/local/bin/snpEff.jar "$@"\n' \
    > /usr/local/bin/snpeff \
 && chmod +x /usr/local/bin/snpeff \
 && rm /tmp/snpEff_latest_core.zip

ENV PATH="/usr/local/bin:/opt/gatk-4.5.0.0:/opt/snpEff:${PATH}"

CMD ["bash"]