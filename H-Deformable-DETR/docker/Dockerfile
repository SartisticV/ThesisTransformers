FROM nvcr.io/nvidia/pytorch:21.03-py3
#FROM nvcr.io/nvidia/pytorch:23.07-py3
# Set noninteractive mode (during build only).
ARG DEBIAN_FRONTEND=noninteractive

# Initialize essential OS-resources.
RUN apt-get update && apt-get -y install apt-utils && apt-get -y upgrade && apt-get install libssl-dev libffi-dev -y && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

# Set internal timezone to Europe/Amsterdam.
RUN apt-get install -y tzdata
RUN ln -s -f /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install ssh server
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
# Fill the /etc/ssh/sshd_config with 5 lines, suppressing ipv6, enabling X11 forwarding, root login and sftp
RUN exec > /etc/ssh/sshd_config &&\
    echo AddressFamily inet &&\
    echo ListenAddress 0.0.0.0 &&\
    echo PermitRootLogin yes &&\
    echo X11Forwarding yes &&\
    echo PermitUserEnvironment yes &&\
    echo Subsystem sftp /usr/lib/openssh/sftp-server
EXPOSE 22

EXPOSE 5151

# Install your favorite commonly used linux tools.
RUN apt-get install -y \
  wget \
  unzip \
  nano \
  htop \
  git

# Install requiremtns for opencv
RUN apt-get install -y \
  ffmpeg \
  libsm6 \
  libxext6

# Optional: remove old opencv python give sometimes problems
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2

# Cleanup OS-resources.
RUN apt-get -y clean

RUN pip install --upgrade pip setuptools wheel

# Project requirements
RUN mkdir -p somewhere
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN FORCE_CUDA=1 pip install mmcv-full==1.3.17 mmdet==2.28.2 timm
RUN pip install fiftyone

# Set you project work dir
WORKDIR /hdef