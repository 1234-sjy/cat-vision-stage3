#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  printf '%s\n' '请用sudo运行此脚本。' >&2
  exit 1
fi
. /etc/os-release
if [ "$ID" != ubuntu ]; then
  printf '%s\n' '此安装脚本适用于Ubuntu；其他Linux请按官方Docker与NVIDIA文档安装。' >&2
  exit 1
fi
apt-get update
apt-get install -y ca-certificates curl gnupg python3
install -m 0755 -d /etc/apt/keyrings
case "${CATVISION_DOCKER_SOURCE:-docker}" in
  docker)
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' \
      "$(dpkg --print-architecture)" "$VERSION_CODENAME" > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ;;
  ubuntu)
    # Alternative signed packages maintained in Ubuntu's official archive.
    apt-get install -y docker.io docker-compose-v2 docker-buildx
    ;;
  *)
    printf '%s\n' 'CATVISION_DOCKER_SOURCE只能是docker或ubuntu。' >&2
    exit 2
    ;;
esac
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
if [ "$(cat /proc/1/comm)" = systemd ]; then
  systemctl enable --now docker
  systemctl restart docker
else
  service docker restart
fi
docker version
docker compose version
printf '%s\n' 'Docker与NVIDIA容器运行时已安装。WSL使用Windows已有的显卡驱动，本脚本不安装Linux显卡内核驱动。'
