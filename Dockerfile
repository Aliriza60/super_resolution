FROM nvidia/cuda:12.1.0-base-ubuntu20.04
RUN apt update
RUN apt install -y git python3 python3-pip wget
WORKDIR /super_resolution
COPY super_resolution.py /super_resolution/
RUN pip install super-image
RUN pip install pillow
CMD ["python","super_resolution.py"]
