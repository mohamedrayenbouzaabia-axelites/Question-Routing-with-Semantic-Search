
#!/bin/bash

# Make sure we have Python 3.8+
python3 --version

# Install dependencies
pip3 install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_openai_api_key_here"
export HOST="0.0.0.0"
export PORT="8000"

# Run the application
python3 -m uvicorn app.main:app --host $HOST --port $PORT --reload