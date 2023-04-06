#!/bin/bash
echo 'starting python server'
# turn on bash's job control
python /app/public/js/swap_faces.py runserver &
sleep 5
echo 'starting nodejs server'
node /app/index.js

echo 'all servers started'
