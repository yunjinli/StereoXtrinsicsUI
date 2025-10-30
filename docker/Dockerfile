FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libglu1-mesa libglut3.12 mesa-utils ffmpeg git \
    tcl8.6 tk8.6 xvfb x11vnc fluxbox websockify novnc x11-apps

WORKDIR /app

COPY requirements.txt .
COPY docker/start_web.sh /usr/local/bin/start_web.sh
RUN chmod +x /usr/local/bin/start_web.sh
EXPOSE 6080

RUN pip install -r requirements.txt

COPY . .

LABEL org.opencontainers.image.title="StereoXtrinsicsUI"
LABEL org.opencontainers.image.description="Interactive GUI for stereo extrinsic calibration"
LABEL org.opencontainers.image.licenses="MIT"

CMD ["python", "UI.py", "--help"]