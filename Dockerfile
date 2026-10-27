FROM osrf/ros:humble-desktop

# Install Ignition Fortress repo
RUN apt-get update && apt-get install -y wget lsb-release gnupg \
    && sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list' \
    && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -

# Install TurtleBot4 simulator + Nav2 + Ignition Fortress
RUN apt-get update && apt-get install -y --no-install-recommends \
    ignition-fortress \
    ros-humble-turtlebot4-simulator \
    ros-humble-irobot-create-nodes \
    ros-humble-turtlebot4-navigation \
    ros-humble-turtlebot4-desktop \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-ros-gz \
    ros-humble-ros-ign-gazebo \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps for our experiments
RUN pip3 install --no-cache-dir numpy scipy pyyaml pillow matplotlib tqdm

# Entry point sources ROS
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
ENV IGNITION_GAZEBO_RESOURCE_PATH=/opt/ros/humble/share

WORKDIR /workspace
