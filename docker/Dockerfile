FROM nvcr.io/nvidia/pytorch:21.10-py3

ARG USER
ARG UID
ARG GID
ARG DEBIAN_FRONTEND=noninteractive

# personally preferred apps
RUN apt-get update && apt-get install -y \
    openssh-server \
    build-essential \
    curl \
    git \
    wget \
    bzip2 \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    software-properties-common \
    unzip \
    zip \
    sudo \
    screen \
    tmux

# required apps for this project
RUN apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libsndfile-dev \
    portaudio19-dev

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${USER}" | chpasswd && adduser ${USER} sudo




###### ssh setup
RUN mkdir /var/run/sshd
# Set root password for SSH access
RUN echo 'root:root' | chpasswd # 'username:userpasswd'
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22
#CMD ["/usr/sbin/sshd", "-D"] #TODO: this doesn't start running ssh service. why?

# Switch to the root user to run conda init and modify the non-root user's bashrc
USER root
###### conda setup
RUN /opt/conda/bin/conda init bash
COPY requirements-cu11.txt /home/${USER}/requirements-cu11.txt
# Ensure the non-root user's .bashrc file activates the conda environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/${USER}/.bashrc && \
    echo "conda activate ai8x-training" >> /home/${USER}/.bashrc

# Set user and workdir
USER ${UID}:${GID}
WORKDIR /home/${USER}


# Copy the requirements file into the Docker image
COPY requirements-cu11.txt /home/${USER}/requirements-cu11.txt

# Create conda env
RUN conda create -y --name ai8x-training python=3.8.12

# Use the environment for pip installations
# Activate the Conda environment explicitly for the command where you need it
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
                  conda activate ai8x-training && \
                  pip install --no-cache-dir -U pip && \
                  pip install gpustat matplotlib tqdm pandas && \
                  pip install -U pip wheel setuptools && \
                  pip install -r /home/${USER}/requirements-cu11.txt && \
                  pip install -e git+https://github.com/MaximIntegratedAI/distiller.git@ea4436d#egg=distiller"

