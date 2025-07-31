# Agent System Setup and Usage Guide

## 🚀 Quick Start

### Option 1: Use the Setup Script (Recommended)
```bash
# Navigate to the agents directory
cd /path/to/agent_s2_project/android_world/android_world/qa_team_agents

# Run the setup script (this will open a new shell with everything configured)
./setup_env.sh
```

### Option 2: Manual Activation
```bash
# Navigate to the agents directory
cd /path/to/agent_s2_project/android_world/android_world/qa_team_agents

# Activate the environment
source activate_env.sh
```

## 📋 Available Commands

Once the environment is activated, you can run:

### Main Orchestrator
```bash
# Show help
python3 main_agent_orchestrator.py --help

# Run with a specific goal
python3 main_agent_orchestrator.py --goal "Turn on Wi-Fi from Settings"

# Run in interactive mode
python3 main_agent_orchestrator.py --interactive

# Run with a task file
python3 main_agent_orchestrator.py --task_file example_task.json
```

### Example Runner
```bash
# Run basic examples
python3 run_example.py

# Run with specific examples
python3 run_example.py --basic
python3 run_example.py --task_file
python3 run_example.py --interactive
```

## 🔧 Environment Details

The setup scripts automatically:
- ✅ Activate the `android_env_3_11` virtual environment
- ✅ Set up Python paths for `android_world` and `Agent-S` modules
- ✅ Install all required dependencies
- ✅ Configure the environment for agent execution

## 📁 File Structure
```
qa_team_agents/
├── main_agent_orchestrator.py    # Main orchestrator
├── Planner_Agent.py              # Planner agent
├── Simple_Executor_With_Verifier.py  # Executor with verifier
├── Supervisor_Agent.py           # Supervisor agent
├── setup_env.sh                  # Setup script
├── activate_env.sh               # Activation script
├── run_example.py                # Example runner
├── example_task.json             # Sample task file
└── README_AGENT_SETUP.md         # This file
```

## 🐛 Troubleshooting

### If you get "command not found: python"
- Make sure you're using `python3` instead of `python`
- Ensure the virtual environment is activated (you should see `(android_env_3_11)` in your prompt)

### If you get import errors
- Run `source activate_env.sh` to set up the Python paths
- Make sure you're in the correct directory

### If you get dependency errors
- The setup script should handle all dependencies automatically
- If issues persist, try: `pip3 install --upgrade --force-reinstall <package_name>`

## 💡 Tips

1. **Always activate the environment first** using one of the setup scripts
2. **Use `python3`** instead of `python` for all commands
3. **Check your prompt** - you should see `(android_env_3_11)` when the environment is active
4. **Use `deactivate`** to exit the virtual environment when done

## 🎯 Example Workflow

```bash
# 1. Navigate to the agents directory
cd /path/to/agent_s2_project/android_world/android_world/qa_team_agents

# 2. Activate the environment
source activate_env.sh

# 3. Run your agents
python3 main_agent_orchestrator.py --goal "Turn on Wi-Fi from Settings"

# 4. When done, deactivate
deactivate
``` 