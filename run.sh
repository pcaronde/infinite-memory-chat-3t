#!/bin/bash

# Infinite Memory Chat - Setup and Run Script

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠️  OPENAI_API_KEY not set!"
    echo "   Run: export OPENAI_API_KEY='your-key'"
    echo "   Then run this script again."
    exit 1
fi

# Run the chat
echo ""
python3 infinite_memory_chat.py
