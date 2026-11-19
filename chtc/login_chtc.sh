#!/bin/zsh
if [ -f .env ]; then
    set -a          # export all variables
    source .env
    set +a
else
    echo "Error: .env file missing"
    exit 1
fi

USER=${CHTC_USER}
HOSTNAME="ap2001.chtc.wisc.edu"
echo "Attempting to login as: ${USER} to ${HOSTNAME}"
ssh ${USER}@${HOSTNAME}