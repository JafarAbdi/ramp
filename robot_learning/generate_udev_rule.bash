#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 /dev/ttyUSB0 desired_symlink_name"
    exit 1
fi

VENDOR=$(udevadm info -n "$1" -q property | grep ID_VENDOR_ID | cut -d= -f2)
PRODUCT=$(udevadm info -n "$1" -q property | grep ID_MODEL_ID | cut -d= -f2)
SERIAL=$(udevadm info -n "$1" -q property | grep ID_SERIAL_SHORT | cut -d= -f2)

echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$VENDOR\", ATTRS{idProduct}==\"$PRODUCT\", ATTRS{serial}==\"$SERIAL\", MODE=\"0666\", SYMLINK+=\"$2\" ATTR{device/latency_timer}=\"1\""
