FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG VLLM_VERSION=0.7.3
ARG VLLM_CUDA_VERSION=121
ARG TORCH_VERSION=2.5.1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /q2d

RUN apt-get update \
    && apt install -y wget gpg software-properties-common lsb-release \
    && mkdir /run/sshd \
    && add-apt-repository 'ppa:deadsnakes/ppa' -y \
    && apt-get update \
    && apt install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dbg \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    curl \
    openssh-server \
    tar \
    wget \
    build-essential \
    cmake \
    git \
    libffi-dev \
    screen \
    htop \
    nano \
    vim \
    fakeroot \
    ncurses-dev \
    xz-utils \
    libssl-dev \
    bc \
    flex \
    libelf-dev \
    bison \
    bear \
    mc \
    libaio-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSk https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && sed -i 's/^#\(PermitRootLogin\) .*/\1 yes/' /etc/ssh/sshd_config \
    && sed -i 's/^\(UsePAM yes\)/# \1/' /etc/ssh/sshd_config

RUN python${PYTHON_VERSION} -m pip install setuptools \
    && python${PYTHON_VERSION} -m pip install torch==${TORCH_VERSION} vllm==${VLLM_VERSION} --trusted-host=github.com --trusted-host=objects.githubusercontent.com --extra-index-url https://download.pytorch.org/whl/cu${VLLM_CUDA_VERSION}

COPY ./requirements.txt /q2d
RUN apt-get remove python3-blinker -y
RUN --mount=type=cache,target=/root/.cache \
    python${PYTHON_VERSION} -m pip install -r requirements.txt

COPY ./q2d /q2d/q2d
COPY ./setup.py /q2d
COPY ./pyproject.toml /q2d
COPY ./scripts /q2d/scripts
RUN chmod +x /q2d/scripts/start.sh
RUN python${PYTHON_VERSION} -m pip install -e .


CMD /q2d/scripts/start.sh
EXPOSE 22
