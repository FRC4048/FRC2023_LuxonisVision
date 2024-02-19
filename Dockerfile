
#
# Docker image file for the Luxonis camera
#
FROM luxonis/depthai-library:656c2f36b9f93df961da81619125cbfee2b651a9

# Default value for network tables server
ENV NT_IP=10.40.48.2

ENV DEBUG=False

ENV SCREEN_FRAME=False

WORKDIR luxonis
# copy model and python scripts
COPY YOLOv8n.json YOLOv8n_openvino_2022.1_6shave.blob networktable_spatial_tiny_yolo.py ./

# Install Network tables library
RUN pip3 install --find-links https://tortall.net/~robotpy/wheels/2023/raspbian/ pyntcore

# Parameters and command to run when the container is run
CMD ["python3", "networktable_spatial_tiny_yolo.py"]
