#!/usr/bin/env python3
"""
Example usage of the Agent Orchestrator
=======================================

This script demonstrates how to use the main orchestrator programmatically.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_agent_orchestrator import AgentOrchestrator

def run_basic_example():
    """Run a basic example with a simple goal."""
    print("🚀 Running Basic Example")
    print("=" * 40)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        goal="Access Wi-Fi settings on Android device",
        visual_trace_dir="example_visual_traces"
    )
    
    # Run the complete workflow
    results = orchestrator.run_complete_workflow()
    
    if 'error' not in results:
        print("\n✅ Example completed successfully!")
        print(f"Episode ID: {results.get('episode_id')}")
        print(f"Subtasks generated: {results.get('planning', {}).get('subtasks_generated', 0)}")
    else:
        print(f"\n❌ Example failed: {results.get('error')}")

def run_with_task_file():
    """Run example using a task file."""
    print("\n🚀 Running Example with Task File")
    print("=" * 40)
    
    # Initialize orchestrator with task file
    orchestrator = AgentOrchestrator(
        task_file="example_task.json",
        visual_trace_dir="task_file_visual_traces"
    )
    
    # Run the complete workflow
    results = orchestrator.run_complete_workflow()
    
    if 'error' not in results:
        print("\n✅ Task file example completed successfully!")
        print(f"Episode ID: {results.get('episode_id')}")
        print(f"Goal: {results.get('goal')}")
    else:
        print(f"\n❌ Task file example failed: {results.get('error')}")

def run_interactive_example():
    """Run interactive mode example."""
    print("\n🚀 Running Interactive Example")
    print("=" * 40)
    print("This will start interactive mode. Type 'help' for commands, 'quit' to exit.")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        goal="Interactive testing",
        visual_trace_dir="interactive_visual_traces"
    )
    
    # Run interactive mode
    orchestrator.run_interactive_mode()

def main():
    """Main function to run examples."""
    print("🎯 Agent Orchestrator Examples")
    print("=" * 50)
    
    # Check if all required files exist
    required_files = [
        "Planner_Agent.py",
        "Simple_Executor_With_Verifier.py", 
        "Supervisor_Agent.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Make sure all agent files are in the same directory")
        return
    
    print("✅ All required files found")
    
    # Run examples
    try:
        # Example 1: Basic usage
        run_basic_example()
        
        # Example 2: Task file usage
        if os.path.exists("example_task.json"):
            run_with_task_file()
        else:
            print("\n⚠️ Skipping task file example (example_task.json not found)")
        
        # Example 3: Interactive mode
        print("\n" + "=" * 50)
        choice = input("Run interactive example? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            run_interactive_example()
        
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 