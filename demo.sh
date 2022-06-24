#!/bin/bash
PORT=5000

pip install -r requirements.txt
echo "Starting Concept Map Recommener at $PORT"
echo "See for documentation: http://localhost:$PORT/"

export FLASK_APP=Webservice.py
export FLASK_ENV=production
flask run --host=0.0.0.0 --no-debugger --port=$PORT
