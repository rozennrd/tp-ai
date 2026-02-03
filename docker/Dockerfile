FROM ubuntu:24.04


ARG UNAME=docker
ARG UID=1000
ARG GID=1000

RUN echo "UID is $UID and GID is $GID"


ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

ENV LC_ALL=C

ENV LIMIT=""

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN userdel -r $(id -un 1000)
RUN if id -u $UID >/dev/null 2>&1; then userdel -r $(id -un $UID); fi

#Install the packages
RUN apt-get $LIMIT update -yq 
RUN apt-get $LIMIT install -y locales locales-all -yq 
RUN apt-get $LIMIT install ssh -yq 
RUN apt-get $LIMIT install x11-apps -yq 
RUN apt-get $LIMIT install sudo -yq 
RUN apt-get $LIMIT install fdisk -yq 
RUN apt-get $LIMIT install gparted -yq 
RUN apt-get $LIMIT install net-tools -yq 
RUN apt-get $LIMIT install iputils-ping -yq 
RUN apt-get $LIMIT install iptables -yq 
RUN apt-get $LIMIT install net-tools -yq
RUN apt-get $LIMIT install iputils-ping -yq 
RUN apt-get $LIMIT install iptables -yq 
RUN apt-get $LIMIT install nano -yq 
RUN apt-get  $LIMIT install nvidia-cuda-toolkit -yq 
RUN apt-get  $LIMIT install git -yq 
RUN apt-get  $LIMIT install python3-pip -yq 
RUN apt-get  $LIMIT install python3-tk -yq 
RUN apt-get  $LIMIT install libudev-dev libsystemd-dev libdrm-dev -yq 
RUN apt-get  $LIMIT install cmake libncurses5-dev libncursesw5-dev -yq 
RUN apt $LIMIT install libcanberra-gtk-module libcanberra-gtk3-module -yq 
RUN apt $LIMIT install python3-numpy -yq
RUN apt $LIMIT install python3-pybind11 -yq
RUN apt-get $LIMIT install python3-venv python3-full -yq
RUN apt-get $LIMIT dist-upgrade -yq 
RUN apt-get clean -yq \
&&  rm -rf /var/lib/apt/lists/* 

# Setup user and sudo access
RUN export UNAME=${UNAME} UID=${UID} GID=${GID} && \
    mkdir -p "/home/${UNAME}" && \
    echo "${UNAME}:x:${UID}:${GID}:${UNAME} User,,,:/home/${UNAME}:/bin/bash" >> /etc/passwd && \
    echo "${UNAME}:x:${UID}:" >> /etc/group && \
    # Set user password
    echo "${UNAME}:${UNAME}" | chpasswd && \
    # Setup sudo
    mkdir -p /etc/sudoers.d && \
    echo "${UNAME} ALL=(ALL) ALL" > /etc/sudoers.d/${UNAME} && \
    chmod 0440 /etc/sudoers.d/${UNAME} && \
    # Add to required groups
    usermod -aG sudo ${UNAME} && \
    gpasswd -a ${UNAME} audio && \
    # Set permissions
    chown ${UID}:${GID} -R /home/${UNAME}

WORKDIR /home/${UNAME}/

# Create separate virtual environments
RUN python3 -m venv /home/${UNAME}/venv_tf && \
    python3 -m venv /home/${UNAME}/venv_torch

# Install TensorFlow packages
RUN . /home/${UNAME}/venv_tf/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install Pillow scipy jupyter ipywidgets numpy matplotlib pandas && \
    pip3 install pybind11 Cython h5py && \
    pip3 install tensorflow[and-cuda] tensorflow-datasets tf-keras transformers Pillow && \
    deactivate

# Install PyTorch packages
RUN . /home/${UNAME}/venv_torch/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install Pillow scipy jupyter ipywidgets numpy matplotlib pandas && \
    pip3 install torch torchvision torchaudio  && \
    deactivate
#--index-url https://download.pytorch.org/whl/cu121
# Create activation scripts
RUN echo 'source /home/${UNAME}/venv_tf/bin/activate' > /home/${UNAME}/activate_tf && \
    echo 'source /home/${UNAME}/venv_torch/bin/activate' > /home/${UNAME}/activate_torch && \
    chmod +x /home/${UNAME}/activate_tf /home/${UNAME}/activate_torch

#Install nvtop
RUN git clone https://github.com/Syllo/nvtop.git \
&& mkdir -p nvtop/build && cd nvtop/build \
&& cmake .. \
&& make \
&& make install


#Add folder to the container
#ADD ./Work ./Work
ADD ./atom ./atom
ADD ./atom/.bashrc ./.bashrc

RUN chmod -R 777 /home/${UNAME}/atom

#Install Atom
RUN apt-get $LIMIT update -yq \
&& dpkg -i ./atom/atom-amd64.deb \
|| true \
&& apt --fix-broken install -yq \
&& dpkg -i ./atom/atom-amd64.deb \
&& apt-get clean -yq \
&& rm -rf /var/lib/apt/lists/*
