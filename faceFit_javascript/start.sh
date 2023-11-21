#!/bin/bash
echo 'starting python server'
# turn on bash's job control
python -u /app/public/python/swap_faces.py &
sleep 5
echo 'starting nodejs server'
node /app/index.js

echo 'all servers started'
