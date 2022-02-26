#!/bin/bash
# This script is run to transfer the compiled parameters to the Vivado host machine.
# It also executes the script which runs Vivado at the end of the script

# 1st arg - Path to file transfer directory (project build sources)
# 2nd arg - SSH call string e.g. user:x.x.x.x
# 3rd arg - Path to Vivado executable on server
# 4th arg - Path to project on host machine.

echo "Beginning parameter file transfer to Vivado host machine..."

# Absolute path this script is in. Will not work for symlinks.
SCRIPTPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CACHEPATH="/file_cache/"
FINALPATH="$SCRIPTPATH$CACHEPATH"

# Transfer the files over SSH to the host machine
rsync -a -v --stats --progress $FINALPATH $2:$1

# Run the Vivado execution script on the host machine; pass the necessary filepaths as arguments
NEXTSCRIPTPATH="${SCRIPTPATH}/run_vivado.sh"
ssh $2 "bash -s" < $NEXTSCRIPTPATH $3 $4 