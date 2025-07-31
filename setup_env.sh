#!/bin/bash

# Agent System Environment Setup Script
# This script activates the virtual environment and sets up Python paths

echo "🚀 Setting up Agent System Environment..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the script directory
cd "$SCRIPT_DIR"

# Activate the virtual environment
echo "📦 Activating virtual environment..."
source ../../../android_world/android_env_3_11/bin/activate

# Set Python paths
echo "🔧 Setting Python paths..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../../android_world"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../../../Agent-S"

# Verify the setup
echo "✅ Environment setup complete!"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python: $(which python3)"
echo "📦 Virtual environment: $VIRTUAL_ENV"

# Show available commands
echo ""
echo "🎯 Available commands:"
echo "  python3 main_agent_orchestrator.py --help"
echo "  python3 main_agent_orchestrator.py --goal 'your goal'"
echo "  python3 main_agent_orchestrator.py --interactive"
echo "  python3 run_example.py"
echo ""

# Keep the shell open with the environment activated
echo "💡 The environment is now ready. You can run your agents!"
echo "💡 To exit this environment, type 'deactivate'"
echo ""

# Start a new shell with the environment
exec $SHELL 