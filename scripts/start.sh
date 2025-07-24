#!/usr/bin/env bash

env | sed 's/^/export /' > /etc/profile

if [ "${SSHD}" == "true" ]; then
  /etc/init.d/ssh start
  sleep infinity
fi
