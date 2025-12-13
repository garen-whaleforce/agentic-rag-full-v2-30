#!/usr/bin/env bash

KEY="$HOME/Downloads/server.pem"
USER_HOST="ubuntu@ec2-3-115-78-26.ap-northeast-1.compute.amazonaws.com"

while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') Starting SSH tunnels..."

  ssh -i "$KEY" \
      -L 15432:localhost:5432 \
      -L 17687:localhost:7687 \
      -L 17474:localhost:7474 \
      -o ServerAliveInterval=60 \
      -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -o TCPKeepAlive=yes \
      "$USER_HOST"

  EXIT_CODE=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') SSH exited with code ${EXIT_CODE}. Reconnecting in 5 seconds..."
  sleep 5
done
