FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ros-humble-realsense2-camera \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rviz2 \
    ros-humble-cv-bridge \
    ros-humble-rqt-image-view \
    python3-opencv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir rosbags

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
WORKDIR /ws
CMD ["bash"]

RUN apt-get update && apt-get install -y nano sudo ros-humble-vision-msgs && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "numpy<2"

ARG USERNAME=dev
RUN useradd --uid 1000 -m -s /bin/bash $USERNAME \
 && usermod -aG video,plugdev $USERNAME \
 && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
 && echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc \
 && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /home/$USERNAME/.bashrc
USER $USERNAME
