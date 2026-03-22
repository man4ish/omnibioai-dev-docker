"""
tests/test_dev_docker.py

Unit tests for omnibioai-dev-docker.

Validates:
  1. Dockerfile structure and required instructions
  2. Required tools and packages are installed
  3. Exposed ports are correct
  4. requirements.txt is valid and complete
  5. run_ai_dev.sh script correctness
  6. Environment variables and labels
  7. Security and best practices
"""

from __future__ import annotations

import re
import shlex
import unittest
from pathlib import Path

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
RUN_SCRIPT = REPO_ROOT / "run_ai_dev.sh"

# ── Expected configuration ────────────────────────────────────────────────────
EXPECTED_BASE_IMAGE = "nvcr.io/nvidia/pytorch"
EXPECTED_PORTS = {3306, 8888, 11434}
EXPECTED_WORKDIR = "/workspace"

# System packages that must be installed
REQUIRED_APT_PACKAGES = {
    "git", "curl", "wget", "build-essential",
    "libssl-dev", "libffi-dev",
    "r-base",
    "fastqc", "samtools", "bcftools",
    "openjdk-11-jre-headless",
}

# Python packages that must be in pip install
REQUIRED_PIP_PACKAGES = {
    "jupyterlab", "notebook",
    "pandas", "numpy", "scipy", "scikit-learn",
    "matplotlib", "seaborn",
    "sqlalchemy", "pymysql",
    "xgboost", "lightgbm", "plotly", "bokeh", "polars",
    "transformers", "datasets", "accelerate",
    "huggingface-hub", "safetensors",
}

# Bioinformatics tools installed via curl/wget
REQUIRED_TOOLS = {"nextflow", "gatk"}

# R packages that must be installed
REQUIRED_R_PACKAGES = {
    "tidyverse", "data.table", "BiocManager",
    "ComplexHeatmap", "limma", "edgeR", "DESeq2",
}

# Docker labels that must be present
REQUIRED_LABELS = {
    "org.opencontainers.image.authors",
    "org.opencontainers.image.description",
    "org.opencontainers.image.version",
}

# run_ai_dev.sh must contain these docker run flags
REQUIRED_DOCKER_FLAGS = {
    "--gpus all",
    "--ipc=host",
    "--ulimit memlock=-1",
}

# Ports that must be published in run script
REQUIRED_PUBLISHED_PORTS = {8888, 11434}


# ── Helpers ───────────────────────────────────────────────────────────────────

def dockerfile_content() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def requirements_content() -> str:
    return REQUIREMENTS.read_text(encoding="utf-8")


def run_script_content() -> str:
    return RUN_SCRIPT.read_text(encoding="utf-8")


def dockerfile_lines() -> list[str]:
    return dockerfile_content().splitlines()


def requirements_packages() -> set[str]:
    """Parse requirements.txt and return lowercase package names."""
    pkgs = set()
    for line in requirements_content().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle various formats: pkg==1.0, pkg>=1.0, pkg @ file://...
        name = re.split(r"[>=<!@\s]", line)[0].strip().lower().replace("-", "_")
        if name:
            pkgs.add(name)
    return pkgs


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFilePresence(unittest.TestCase):
    """All required files must exist."""

    def test_dockerfile_exists(self) -> None:
        self.assertTrue(DOCKERFILE.exists(), "Dockerfile not found")

    def test_requirements_exists(self) -> None:
        self.assertTrue(REQUIREMENTS.exists(), "requirements.txt not found")

    def test_run_script_exists(self) -> None:
        self.assertTrue(RUN_SCRIPT.exists(), "run_ai_dev.sh not found")

    def test_readme_exists(self) -> None:
        readme = REPO_ROOT / "README.md"
        self.assertTrue(readme.exists(), "README.md not found")

    def test_dockerfile_not_empty(self) -> None:
        self.assertGreater(DOCKERFILE.stat().st_size, 0, "Dockerfile is empty")

    def test_requirements_not_empty(self) -> None:
        self.assertGreater(REQUIREMENTS.stat().st_size, 0, "requirements.txt is empty")

    def test_run_script_not_empty(self) -> None:
        self.assertGreater(RUN_SCRIPT.stat().st_size, 0, "run_ai_dev.sh is empty")


class TestDockerfileBaseImage(unittest.TestCase):
    """Dockerfile must use the correct NVIDIA base image."""

    def test_from_instruction_present(self) -> None:
        content = dockerfile_content()
        self.assertIn("FROM ", content, "No FROM instruction found")

    def test_base_image_is_nvidia_pytorch(self) -> None:
        content = dockerfile_content()
        from_lines = [l for l in dockerfile_lines() if l.startswith("FROM ")]
        self.assertTrue(len(from_lines) > 0, "No FROM instruction")
        self.assertIn(
            EXPECTED_BASE_IMAGE,
            from_lines[0],
            f"Base image should be {EXPECTED_BASE_IMAGE}",
        )

    def test_base_image_has_cuda_tag(self) -> None:
        content = dockerfile_content()
        from_lines = [l for l in dockerfile_lines() if l.startswith("FROM ")]
        # Tag should contain py3 (NVIDIA pytorch convention)
        self.assertIn("py3", from_lines[0], "Base image tag should include 'py3'")


class TestDockerfileLabels(unittest.TestCase):
    """Required LABEL instructions must be present."""

    def test_required_labels_present(self) -> None:
        content = dockerfile_content()
        errors = []
        for label in REQUIRED_LABELS:
            if label not in content:
                errors.append(f"Missing LABEL: {label}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_maintainer_label_not_empty(self) -> None:
        content = dockerfile_content()
        match = re.search(r'LABEL org\.opencontainers\.image\.authors="([^"]+)"', content)
        self.assertIsNotNone(match, "authors label not found or empty")
        self.assertTrue(match.group(1).strip(), "authors label is empty")

    def test_version_label_follows_semver(self) -> None:
        content = dockerfile_content()
        match = re.search(r'LABEL org\.opencontainers\.image\.version="([^"]+)"', content)
        self.assertIsNotNone(match, "version label not found")
        version = match.group(1)
        self.assertRegex(
            version,
            r"^\d+\.\d+\.\d+$",
            f"version '{version}' does not follow semver",
        )


class TestDockerfileWorkdir(unittest.TestCase):
    """WORKDIR must be set correctly."""

    def test_workdir_is_set(self) -> None:
        content = dockerfile_content()
        self.assertIn("WORKDIR", content, "No WORKDIR instruction")

    def test_workdir_is_workspace(self) -> None:
        content = dockerfile_content()
        self.assertIn(
            f"WORKDIR {EXPECTED_WORKDIR}",
            content,
            f"WORKDIR should be {EXPECTED_WORKDIR}",
        )


class TestDockerfilePorts(unittest.TestCase):
    """All required ports must be exposed."""

    def _exposed_ports(self) -> set[int]:
        ports = set()
        for line in dockerfile_lines():
            if line.startswith("EXPOSE "):
                for port in line.replace("EXPOSE", "").split():
                    try:
                        ports.add(int(port.strip()))
                    except ValueError:
                        pass
        return ports

    def test_all_required_ports_exposed(self) -> None:
        exposed = self._exposed_ports()
        for port in EXPECTED_PORTS:
            self.assertIn(port, exposed, f"Port {port} not exposed")

    def test_mysql_port_exposed(self) -> None:
        self.assertIn(3306, self._exposed_ports())

    def test_jupyter_port_exposed(self) -> None:
        self.assertIn(8888, self._exposed_ports())

    def test_ollama_port_exposed(self) -> None:
        self.assertIn(11434, self._exposed_ports())


class TestDockerfileAptPackages(unittest.TestCase):
    """Required apt packages must be in RUN apt-get install."""

    def test_required_apt_packages_installed(self) -> None:
        content = dockerfile_content()
        errors = []
        for pkg in REQUIRED_APT_PACKAGES:
            if pkg not in content:
                errors.append(f"apt package not found: {pkg}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_apt_cache_cleaned(self) -> None:
        content = dockerfile_content()
        self.assertIn(
            "rm -rf /var/lib/apt/lists",
            content,
            "apt cache not cleaned — adds unnecessary image size",
        )

    def test_apt_install_uses_no_install_recommends(self) -> None:
        content = dockerfile_content()
        self.assertIn(
            "--no-install-recommends",
            content,
            "apt-get install should use --no-install-recommends",
        )


class TestDockerfilePipPackages(unittest.TestCase):
    """Required Python packages must be in pip install."""

    def test_required_pip_packages_installed(self) -> None:
        # Packages installed via -r requirements.txt — check requirements.txt instead
        req_content = requirements_content().lower()
        dockerfile = dockerfile_content()
        # Verify Dockerfile references requirements.txt
        self.assertIn(
            "requirements.txt",
            dockerfile,
            "Dockerfile should install from requirements.txt",
        )
        errors = []
        for pkg in REQUIRED_PIP_PACKAGES:
            pkg_norm = pkg.lower().replace("-", "_").replace("_", "-")
            pkg_alt = pkg.lower().replace("-", "_")
            if pkg_norm not in req_content and pkg_alt not in req_content and pkg.lower() not in req_content:
                errors.append(f"pip package not found in requirements.txt: {pkg}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_pip_upgraded_before_install(self) -> None:
        content = dockerfile_content()
        # Accept either --upgrade pip or --upgrade as part of install command
        self.assertTrue(
            "pip install --upgrade pip" in content or
            "pip install --no-cache-dir --upgrade pip" in content or
            "--upgrade pip" in content,
            "pip should be upgraded before installing packages",
        )


class TestDockerfileBioinformaticsTools(unittest.TestCase):
    """Bioinformatics tools must be installed."""

    def test_nextflow_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("nextflow", content.lower(), "Nextflow not installed")

    def test_gatk_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("gatk", content.lower(), "GATK not installed")

    def test_fastqc_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("fastqc", content.lower(), "FastQC not installed")

    def test_samtools_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("samtools", content, "SAMtools not installed")

    def test_bcftools_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("bcftools", content, "BCFtools not installed")

    def test_snpeff_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("snpEff", content, "SnpEff not installed")

    def test_ollama_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("ollama", content.lower(), "Ollama not installed")

    def test_java_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("openjdk", content, "Java (OpenJDK) not installed")


class TestDockerfileRPackages(unittest.TestCase):
    """R and Bioconductor packages must be installed."""

    def test_r_base_installed(self) -> None:
        content = dockerfile_content()
        self.assertIn("r-base", content, "r-base not installed")

    def test_required_r_packages_present(self) -> None:
        content = dockerfile_content()
        errors = []
        for pkg in REQUIRED_R_PACKAGES:
            if pkg not in content:
                errors.append(f"R package not found: {pkg}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_biocmanager_used(self) -> None:
        content = dockerfile_content()
        self.assertIn("BiocManager", content, "BiocManager not used for Bioconductor packages")


class TestDockerfileJupyter(unittest.TestCase):
    """Jupyter must be configured for remote access."""

    def test_jupyter_config_created(self) -> None:
        content = dockerfile_content()
        self.assertIn("jupyter_notebook_config.py", content)

    def test_jupyter_configured_for_all_interfaces(self) -> None:
        content = dockerfile_content()
        self.assertIn("0.0.0.0", content, "Jupyter not configured to listen on all interfaces")

    def test_jupyter_browser_disabled(self) -> None:
        content = dockerfile_content()
        self.assertIn("open_browser = False", content, "Jupyter browser auto-launch not disabled")

    def test_jupyter_allow_root(self) -> None:
        content = dockerfile_content()
        self.assertIn("allow_root = True", content, "Jupyter root access not enabled")


class TestDockerfilePathEnv(unittest.TestCase):
    """PATH environment variable must include tool directories."""

    def test_path_env_set(self) -> None:
        content = dockerfile_content()
        self.assertIn("ENV PATH=", content, "PATH environment variable not set")

    def test_gatk_in_path(self) -> None:
        content = dockerfile_content()
        self.assertIn("gatk", content.lower())

    def test_snpeff_in_path(self) -> None:
        content = dockerfile_content()
        self.assertIn("snpEff", content)


class TestDockerfileCmd(unittest.TestCase):
    """CMD instruction must be present."""

    def test_cmd_instruction_present(self) -> None:
        content = dockerfile_content()
        self.assertIn("CMD ", content, "No CMD instruction in Dockerfile")

    def test_cmd_is_bash(self) -> None:
        content = dockerfile_content()
        self.assertIn('CMD ["bash"]', content, "CMD should default to bash")


class TestRequirementsTxt(unittest.TestCase):
    """requirements.txt must be valid and contain key packages."""

    REQUIRED_IN_REQUIREMENTS = {
        "torch", "transformers", "datasets", "accelerate",
        "huggingface_hub", "safetensors",
        "jupyterlab", "notebook",
        "pandas", "numpy", "scipy", "scikit_learn",
        "matplotlib", "seaborn",
        "xgboost", "lightgbm", "plotly", "bokeh", "polars",
        "sqlalchemy", "pymysql",
        "pytest",
    }

    def test_requirements_parseable(self) -> None:
        pkgs = requirements_packages()
        self.assertGreater(len(pkgs), 0, "requirements.txt has no packages")

    def test_required_packages_in_requirements(self) -> None:
        pkgs = requirements_packages()
        errors = []
        for pkg in self.REQUIRED_IN_REQUIREMENTS:
            if pkg.lower() not in pkgs:
                errors.append(f"Missing from requirements.txt: {pkg}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_no_duplicate_packages(self) -> None:
        names = []
        for line in REQUIREMENTS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[>=<!@\s]", line)[0].strip().lower()
            if name:
                names.append(name)
        duplicates = [n for n in set(names) if names.count(n) > 1]
        self.assertEqual(duplicates, [], f"Duplicate packages: {duplicates}")

    def test_pytorch_in_requirements(self) -> None:
        content = requirements_content()
        self.assertIn("torch", content.lower(), "PyTorch not in requirements.txt")

    def test_jupyter_in_requirements(self) -> None:
        content = requirements_content()
        self.assertIn("jupyterlab", content.lower())


class TestRunScript(unittest.TestCase):
    """run_ai_dev.sh must be correct and safe."""

    def test_shebang_present(self) -> None:
        content = run_script_content()
        self.assertTrue(
            content.startswith("#!/bin/bash") or content.startswith("#!/usr/bin/env bash"),
            "run_ai_dev.sh missing bash shebang",
        )

    def test_image_name_defined(self) -> None:
        content = run_script_content()
        self.assertIn("IMAGE_NAME=", content, "IMAGE_NAME not defined")

    def test_gpu_flag_present(self) -> None:
        content = run_script_content()
        self.assertIn("--gpus all", content, "GPU flag --gpus all not present")

    def test_ipc_host_flag_present(self) -> None:
        content = run_script_content()
        self.assertIn("--ipc=host", content, "--ipc=host required for PyTorch DataLoader")

    def test_memlock_ulimit_present(self) -> None:
        content = run_script_content()
        self.assertIn("--ulimit memlock=-1", content, "memlock ulimit not set")

    def test_jupyter_port_published(self) -> None:
        content = run_script_content()
        self.assertIn("-p 8888:8888", content, "Jupyter port 8888 not published")

    def test_ollama_port_published(self) -> None:
        content = run_script_content()
        self.assertIn("-p 11434:11434", content, "Ollama port 11434 not published")

    def test_huggingface_cache_mounted(self) -> None:
        content = run_script_content()
        self.assertIn(".cache/huggingface", content, "HuggingFace cache not mounted")

    def test_ollama_model_dir_mounted(self) -> None:
        content = run_script_content()
        self.assertIn(".ollama", content, "Ollama model directory not mounted")

    def test_workspace_mounted(self) -> None:
        content = run_script_content()
        self.assertIn("/workspace", content, "Workspace not mounted")

    def test_hf_home_env_set(self) -> None:
        content = run_script_content()
        self.assertIn("HF_HOME=", content, "HF_HOME env var not set")

    def test_ollama_host_env_set(self) -> None:
        content = run_script_content()
        self.assertIn("OLLAMA_HOST=", content, "OLLAMA_HOST env var not set")

    def test_nvidia_smi_check_present(self) -> None:
        content = run_script_content()
        self.assertIn("nvidia-smi", content, "nvidia-smi check not present")

    def test_container_cleanup_before_run(self) -> None:
        content = run_script_content()
        self.assertIn("docker rm -f", content, "Old container not cleaned up before run")

    def test_interactive_flag_present(self) -> None:
        content = run_script_content()
        self.assertTrue(
            "-it " in content or "-i " in content,
            "Container not started in interactive mode",
        )


class TestDockerfileSecurity(unittest.TestCase):
    """Basic security and best practice checks."""

    def test_no_hardcoded_passwords(self) -> None:
        content = dockerfile_content()
        # Check for common password patterns
        bad_patterns = [
            r'PASSWORD\s*=\s*["\'][^"\']+["\']',
            r'SECRET\s*=\s*["\'][^"\']+["\']',
        ]
        for pattern in bad_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            self.assertIsNone(match, f"Possible hardcoded secret found: {match}")

    def test_no_hardcoded_api_keys(self) -> None:
        content = dockerfile_content()
        self.assertNotRegex(
            content,
            r'hf_[A-Za-z]{20,}',
            "HuggingFace API key appears hardcoded",
        )

    def test_apt_update_before_install(self) -> None:
        content = dockerfile_content()
        # Every apt-get install should be preceded by apt-get update in same RUN
        run_blocks = re.findall(r'RUN.*?(?=\nRUN|\nFROM|\nCMD|\nENTRYPOINT|\Z)',
                                 content, re.DOTALL)
        errors = []
        for block in run_blocks:
            if "apt-get install" in block and "apt-get update" not in block:
                first_line = block.split('\n')[0][:80]
                errors.append(f"apt-get install without update: {first_line}")
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()