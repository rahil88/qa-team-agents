

# QA Team Agents - Android Task Automation System

## 🎯 Project Overview

This project implements a sophisticated multi-agent system for automating Android device tasks through intelligent planning, execution, and verification. The system consists of three main components that work together to achieve complex goals on Android devices.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Planner Agent │───▶│    Executor with     │───▶│ Supervisor Agent│
│                 │    │     Verifier         │    │                 │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
         │                       │                           │
         ▼                       ▼                           ▼
   Generates Plan         Executes Tasks              Records Visual Traces
   Breaks down goal       Verifies each step         Provides Analysis
```

## 🔧 Components

### 1. **Planner Agent** (`Planner_Agent.py`)
- **Purpose**: Breaks down high-level goals into executable subtasks
- **Input**: Natural language goal (e.g., "Access Wi-Fi settings")
- **Output**: Structured list of subtasks with detailed instructions
- **Key Features**:
  - Uses Agent-S Manager for intelligent task decomposition
  - Integrates with Android environment grounding agent
  - Generates human-readable subtask descriptions

### 2. **Executor with Verifier** (`Simple_Executor_With_Verifier.py`)
- **Purpose**: Executes subtasks and verifies each step
- **Input**: List of subtasks from Planner Agent
- **Output**: Execution results with verification status
- **Key Features**:
  - Executes each subtask using Android environment
  - Calls Verifier Agent after each execution
  - Records visual traces for analysis
  - Provides detailed execution logs

### 3. **Supervisor Agent** (`Supervisor_Agent.py`)
- **Purpose**: Records visual traces and provides comprehensive analysis
- **Input**: Execution episodes and visual data
- **Output**: Visual trace recordings and episode summaries
- **Key Features**:
  - Captures screenshots before/after each action
  - Records UI state changes
  - Provides episode-level analysis
  - Stores visual traces for debugging

## 🚀 Quick Start

### Prerequisites

1. **Android SDK** installed and configured
2. **Android Emulator** running on port 5554
3. **Python 3.11** with virtual environment
4. **OpenAI API Key** set as environment variable

### Environment Setup

```bash
# Navigate to the project directory
cd agent_s2_project/android_world/android_world/qa_team_agents

# Set up the environment
source setup_env.sh
activate_env.sh
# Verify setup
python3 --version  # Should show Python 3.11.x
echo $OPENAI_API_KEY  # Should show your API key
```

## 📋 Step-by-Step Execution Flow

## Method-1 Working perfectly 

## All agents have the features as mentioned in code_challenge but still working on issues with main_agent_orchestrator.py!!

### Step 1: Run Planner Agent First

The Planner Agent is responsible for breaking down your goal into executable subtasks.

```bash
#Method 1: Run Planner Agent directly(Working Perfectly With Zero Errors)
#Step1
python3 Planner_Agent.py
#Step2
python3 Simple_Executor_With_Verifier.py

# Method 2: Use the main orchestrator with a goal
python3 main_agent_orchestrator.py --goal "Access Wi-Fi settings and toggle Wi-Fi on/off"

# Method 3: Use a task file
# Runs with Planner agents created subtasks.json if does not find it will take task from example_task.json file

python3 main_agent_orchestrator.py --task_file example_task.json
```

**Expected Output:**
```
✅ Using existing enhanced knowledge base...
🎯 Planning task: Access Wi-Fi settings and toggle Wi-Fi on/off
✅ Generated plan with 5 subtasks
  1. open settings app
  2. wait for settings to load
  3. tap network & internet
  4. tap internet option
  5. toggle wifi switch
```

### Step 2: Run Simple Executor with Verifier

The Simple Executor takes the generated subtasks and executes them in the Android emulator with verification.

```bash
# The executor is automatically called by the main orchestrator
# But you can also run it independently:

python3 Simple_Executor_With_Verifier.py
```

**Expected Output:**
```
🚀 STARTING EXECUTION WITH VERIFICATION
==================================================
🎯 SUBTASK 1/5: open settings app
----------------------------------------
📸 Recording frame before execution...
🔧 Executing: tap on "Settings" app icon
✅ Execution successful
🔍 Verifying execution...
✅ Verification passed: Settings app opened successfully
```

### Step 3: Monitor Supervisor Agent

The Supervisor Agent automatically records visual traces and provides analysis.

```bash
# Supervisor runs automatically, but you can check its output:
ls visual_traces/
# You should see episode directories with screenshots and logs
```

## 🎯 Complete Workflow Example

### Example 1: Wi-Fi Settings Task

```bash
# 1. Set up environment
source setup_env.sh

# 2. Run complete workflow
python3 main_agent_orchestrator.py --task_file example_task.json
```

**What happens:**
1. **Planning Phase**: Planner Agent breaks down "Access Wi-Fi settings" into 5 subtasks
2. **Execution Phase**: Simple Executor runs each subtask in the emulator
3. **Verification Phase**: Verifier Agent checks each step was successful
4. **Recording Phase**: Supervisor Agent captures visual traces
5. **Analysis Phase**: Final verification and summary generation

### Example 2: Interactive Mode

```bash
# Run in interactive mode for testing
python3 main_agent_orchestrator.py --interactive
```

**Interactive Commands:**
- `help`: Show available commands
- `plan`: Generate a plan for a goal
- `analyze`: Show supervisor analysis
- `status`: Show current status
- `quit`: Exit interactive mode

## 📁 File Structure

```
qa_team_agents/
├── README.md                           # This file
├── main_agent_orchestrator.py          # Main orchestrator
├── Planner_Agent.py                    # Task planning agent
├── Simple_Executor_With_Verifier.py    # Task execution with verification
├── Supervisor_Agent.py                 # Visual trace recording
├── Verifier_Agent.py                   # Task verification
├── example_task.json                   # Sample task definition
├── run_example.py                      # Example runner
├── setup_env.sh                        # Environment setup script
├── activate_env.sh                     # Environment activation
├── visual_traces/                      # Visual trace recordings
└── README_AGENT_SETUP.md              # Detailed setup guide
```

## 🔍 Task Definition Format

Tasks are defined in JSON format:

```json
{
  "goal": "Access Wi-Fi settings and toggle Wi-Fi on/off",
  "description": "Navigate to Wi-Fi settings and toggle the Wi-Fi connection",
  "expected_outcome": "Wi-Fi settings should be accessible and Wi-Fi can be toggled",
  "difficulty": "medium",
  "subtasks": [
    {
      "name": "open settings app",
      "info": "Locate and tap the Settings gear icon"
    },
    {
      "name": "tap network & internet",
      "info": "Tap 'Network & internet' option in Settings"
    }
  ],
  "verification_criteria": [
    "Settings app opens successfully",
    "Wi-Fi toggle is visible and functional"
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

1. **"Planner object has no attribute 'generate_plan'"**
   - ✅ **Fixed**: Added `generate_plan` method to Planner_Agent.py

2. **Import errors**
   - Run `source activate_env.sh` to set up Python paths
   - Ensure you're in the correct directory

3. **Android emulator not found**
   - Start Android emulator: `emulator -avd <avd_name> -port 5554`
   - Verify ADB connection: `adb devices`

4. **OpenAI API errors**
   - Set your API key: `export OPENAI_API_KEY="your-key-here"`
   - Check API key validity

### Debug Mode

```bash
# Run with verbose logging
python3 main_agent_orchestrator.py --goal "your goal" --debug

# Check visual traces
ls visual_traces/episode_*/screenshots/
```

## 📊 Output and Results

### Execution Reports
- **Subtask Results**: Individual execution status for each step
- **Verification Logs**: Detailed verification results
- **Visual Traces**: Screenshots and UI state recordings
- **Episode Summary**: Overall task completion status

### Sample Output Structure
```
📋 WORKFLOW SUMMARY:
Episode ID: episode_1732987654
Goal: Access Wi-Fi settings and toggle Wi-Fi on/off
Subtasks: 5
Overall Grade: A
Overall Score: 95.50
```

## 🎓 Learning and Customization

### Adding New Tasks
1. Create a new JSON task file following the format above
2. Define clear subtasks and verification criteria
3. Test with the main orchestrator

### Extending Agents
- **Planner Agent**: Modify task decomposition logic
- **Executor**: Add new action types or UI interactions
- **Verifier**: Enhance verification criteria
- **Supervisor**: Customize visual trace recording

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with the Android emulator
5. Submit a pull request

## 📄 License

This project is part of the QA Team Agents coding challenge.

---

**Note**: This system requires an active Android emulator and OpenAI API access to function properly. Ensure all prerequisites are met before running the agents. 
