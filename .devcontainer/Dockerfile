# Adapted from https://github.com/bstollnitz/fashion-mnist/blob/main/.devcontainer/Dockerfile
# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/python-3-miniconda/.devcontainer/base.Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Update the conda environment according to the conda.yml file in the project.
COPY safe-driver/environ/conda-environment.yml /tmp/conda-tmp/
RUN /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/conda-environment.yml && rm -rf /tmp/conda-tmp

# Add dev specific packages
COPY .devcontainer/dev-requirements.txt /tmp/requirements-tmp/
RUN /opt/conda/bin/pip install -r /tmp/requirements-tmp/dev-requirements.txt && rm -rf /tmp/requirements-tmp