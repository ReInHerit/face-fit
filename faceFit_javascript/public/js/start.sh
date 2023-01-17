#!/bin/bash
echo 'starting python sever'
# turn on bash's job control
#set -m
python /app/public/js/swap_faces.py runserver &
#P1=$?
sleep 5
echo 'starting nodejs sever'
node /app/index.js
#P2=$!
#wait $P1 $P2
# now we bring the primary process back into the foreground
# and leave it there
#fg %1
echo 'all servers started'

#
## Wait for any process to exit
#wait -n
##
### Exit with status of process that exited first
#exit $?