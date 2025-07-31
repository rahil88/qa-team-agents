#!/bin/bash

# Simple Agent Environment Activation Script
# Source this script to activate the environment: source activate_env.sh

echo "🚀 Activating Agent System Environment..."

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

echo "✅ Environment activated!"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python: $(which python3)"
echo "📦 Virtual environment: $VIRTUAL_ENV"
echo ""
echo "🎯 You can now run:"
echo "  python3 main_agent_orchestrator.py --help"
echo "  python3 main_agent_orchestrator.py --goal 'your goal'"
echo "  python3 main_agent_orchestrator.py --interactive"
echo "" 