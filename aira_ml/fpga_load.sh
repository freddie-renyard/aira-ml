#!/bin/bash
# This script transfers the bitstream to the device from the receive cache directory.

# Absolute path this script is in. Will not work for symlinks.
SCRIPTPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CACHEPATH="/rx_cache/au_top.bit"
FINALPATH="$SCRIPTPATH$CACHEPATH"

echo $FINALPATH

openFPGALoader -b alchitry_au $FINALPATH # Loading in SRAM (volatile)
