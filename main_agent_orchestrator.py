#!/usr/bin/env python3
"""
Main Agent Orchestrator
=======================

This file orchestrates the complete agent system including:
- planner Agent: Generates action plans
- Simple Executor with Verifier: Executes plans with verification
- Supervisor Agent: Records visual traces and provides analysis

Usage:
    python main_agent_orchestrator.py --goal "your goal here"
    python main_agent_orchestrator.py --task_file tasks.json
    python main_agent_orchestrator.py --interactive
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the agent modules
try:
    from Planner_Agent import Planner
    from Simple_Executor_With_Verifier import SimpleExecutorWithVerifier
    from Supervisor_Agent import SupervisorAgent
    print("✅ All agent modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing agent modules: {e}")
    print("Make sure all agent files are in the same directory")
    sys.exit(1)

class AgentOrchestrator:
    """
    Main orchestrator for the complete agent system.
    Coordinates planner, executor with Verifier, and Supervisor agents.
    """
    
    def __init__(self, 
                 goal: str = None,
                 task_file: str = None,
                 visual_trace_dir: str = "visual_traces",
                 llm_client = None):
        """
        Initialize the agent orchestrator.
        
        Args:
            goal: The main goal to achieve
            task_file: Path to JSON file containing tasks
            visual_trace_dir: Directory for storing visual traces
            llm_client: LLM client for supervisor analysis
        """
        self.goal = goal
        self.task_file = task_file
        self.visual_trace_dir = visual_trace_dir
        self.llm_client = llm_client
        self.task_config = None  # Store loaded task configuration
        
        # Initialize agents
        self.planner = None
        self.executor = None
        self.supervisor = None
        
        # Execution state
        self.episode_id = None
        self.execution_logs = []
        self.verifier_logs = []
        self.final_results = {}
        
        print("🚀 Initializing Agent Orchestrator...")
    
    def initialize_agents(self):
        """Initialize all agent components."""
        try:
            print("📋 Initializing Planner Agent...")
            self.planner = Planner()
            
            print("🔧 Initializing Simple Executor with Verifier...")
            # Create grounding agent for the executor
            from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
            from android_world.env import env_launcher
            
            # Set up Android environment
            env = env_launcher.load_and_setup_env(
                console_port=5554,
                emulator_setup=False,
                freeze_datetime=True,
                adb_path="/Users/craddy-san/Library/Android/sdk/platform-tools/adb",
                grpc_port=8554,
            )
            grounding_agent = AndroidEnvGroundingAgent(env)
            
            self.executor = SimpleExecutorWithVerifier(
                grounding_agent=grounding_agent,
                subtasks=[],  # Will be populated by planner
                episode_id=self.episode_id
            )
            
            print("👁️ Initializing Supervisor Agent...")
            self.supervisor = SupervisorAgent(
                llm_client=self.llm_client,
                visual_trace_dir=self.visual_trace_dir
            )
            
            print("✅ All agents initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing agents: {e}")
            traceback.print_exc()
            return False
    
    def load_task_from_file(self, task_file: str) -> Dict[str, Any]:
        """Load task configuration from JSON file."""
        try:
            with open(task_file, 'r') as f:
                task_config = json.load(f)
            
            print(f"📄 Loaded task configuration from {task_file}")
            return task_config
            
        except Exception as e:
            print(f"❌ Error loading task file {task_file}: {e}")
            return None
    
    def generate_episode_id(self) -> str:
        """Generate a unique episode ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_id = f"episode_{timestamp}"
        return self.episode_id
    
    def plan_task(self, goal: str, current_state: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Use the planner agent to generate a plan for the given goal.
        Priority order: subtasks.json > task file > planner agent
        
        Args:
            goal: The goal to achieve
            current_state: Current state information
            
        Returns:
            List of subtasks to execute
        """
        try:
            print(f"🎯 Planning task: {goal}")
            
            # First, check if subtasks.json exists (from Planner Agent)
            subtasks_file = "subtasks.json"
            if os.path.exists(subtasks_file):
                try:
                    with open(subtasks_file, 'r') as f:
                        subtasks = json.load(f)
                    print(f"📋 Using subtasks from {subtasks_file} (generated by Planner Agent)")
                    print(f"✅ Using {len(subtasks)} subtasks from Planner Agent:")
                    for i, subtask in enumerate(subtasks):
                        print(f"  {i+1}. {subtask.get('name', 'Unknown task')}")
                    return subtasks
                except Exception as e:
                    print(f"⚠️ Error reading {subtasks_file}: {e}")
            
            # Second, check if we have predefined subtasks from task file
            if self.task_config and 'subtasks' in self.task_config:
                print("📋 Using predefined subtasks from task file")
                subtasks = self.task_config['subtasks']
                print(f"✅ Using {len(subtasks)} predefined subtasks:")
                for i, subtask in enumerate(subtasks):
                    print(f"  {i+1}. {subtask.get('name', 'Unknown task')}")
                return subtasks
            
            # Third, fall back to planner agent if no subtasks available
            if not self.planner:
                print("❌ planner agent not initialized")
                return []
            
            print("🤖 Generating subtasks using Planner Agent")
            # Generate plan using the planner
            plan = self.planner.generate_plan(goal, current_state)
            
            if plan and isinstance(plan, list):
                print(f"✅ Generated plan with {len(plan)} subtasks")
                for i, subtask in enumerate(plan):
                    print(f"  {i+1}. {subtask.get('name', 'Unknown task')}")
                return plan
            else:
                print("❌ Failed to generate valid plan")
                return []
                
        except Exception as e:
            print(f"❌ Error during planning: {e}")
            traceback.print_exc()
            return []
    
    def execute_plan(self, subtasks: List[Dict[str, Any]], env = None) -> Dict[str, Any]:
        """
        Execute the plan using the Simple Executor with Verifier.
        
        Args:
            subtasks: List of subtasks to execute
            env: Environment object for execution
            
        Returns:
            Execution results
        """
        try:
            print(f"🚀 Executing plan with {len(subtasks)} subtasks")
            
            if not self.executor:
                print("❌ Executor agent not initialized")
                return {'error': 'Executor not initialized'}
            
            # Update executor with new subtasks
            self.executor.subtasks = subtasks
            
            # Execute the plan
            results = self.executor.execute_subtasks_with_verification(
                goal=self.goal
            )
            
            print("✅ Plan execution completed")
            return results
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            traceback.print_exc()
            return {'error': str(e)}
    
    def run_complete_workflow(self, env = None) -> Dict[str, Any]:
        """
        Run the complete workflow: Plan -> Execute -> Analyze.
        
        Args:
            env: Environment object for execution
            
        Returns:
            Complete workflow results
        """
        try:
            print("🔄 Starting complete agent workflow...")
            
            # Generate episode ID
            episode_id = self.generate_episode_id()
            print(f"📝 Episode ID: {episode_id}")
            
            # Initialize agents
            if not self.initialize_agents():
                return {'error': 'Failed to initialize agents'}
            
            # Start supervisor recording
            print("📸 Starting supervisor visual trace recording...")
            self.supervisor.start_episode_recording(episode_id, self.goal)
            
            # Phase 1: Planning
            print("\n" + "="*50)
            print("PHASE 1: PLANNING")
            print("="*50)
            
            subtasks = self.plan_task(self.goal)
            if not subtasks:
                return {'error': 'Failed to generate plan'}
            
            # Phase 2: Execution
            print("\n" + "="*50)
            print("PHASE 2: EXECUTION")
            print("="*50)
            
            execution_results = self.execute_plan(subtasks, env)
            if 'error' in execution_results:
                return execution_results
            
            # Phase 3: Analysis
            print("\n" + "="*50)
            print("PHASE 3: ANALYSIS")
            print("="*50)
            
            # End supervisor recording
            print("📸 Ending supervisor visual trace recording...")
            episode_summary = self.supervisor.end_episode_recording("completed")
            
            # Analyze the episode
            print("🔍 Analyzing episode with supervisor...")
            analysis_results = self.supervisor.analyze_episode()
            
            # Generate evaluation report
            print("📊 Generating evaluation report...")
            evaluation_report = self.supervisor.create_evaluation_report()
            
            # Compile final results
            final_results = {
                'episode_id': episode_id,
                'goal': self.goal,
                'planning': {
                    'subtasks_generated': len(subtasks),
                    'subtasks': subtasks
                },
                'execution': execution_results,
                'analysis': analysis_results,
                'evaluation': evaluation_report,
                'supervisor_summary': self.supervisor.get_supervisor_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save complete results
            self.save_results(final_results)
            
            print("\n" + "="*50)
            print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*50)
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error in complete workflow: {e}")
            traceback.print_exc()
            
            # Try to end supervisor recording if it was started
            if self.supervisor and self.episode_id:
                try:
                    self.supervisor.end_episode_recording("failed")
                except:
                    pass
            
            return {'error': str(e)}
    
    def save_results(self, results: Dict[str, Any]):
        """Save the complete results to a JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_workflow_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save results: {e}")
    
    def run_interactive_mode(self):
        """Run the orchestrator in interactive mode."""
        print("🎮 Starting interactive mode...")
        print("Type 'help' for available commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n🤖 Agent Orchestrator > ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    print("👋 Goodbye!")
                    break
                elif command == 'help':
                    self.show_interactive_help()
                elif command == 'plan':
                    goal = input("Enter goal: ")
                    subtasks = self.plan_task(goal)
                    if subtasks:
                        print(f"Generated {len(subtasks)} subtasks")
                elif command == 'analyze':
                    if self.supervisor:
                        summary = self.supervisor.get_supervisor_summary()
                        print(f"Supervisor Summary: {summary}")
                    else:
                        print("Supervisor not initialized")
                elif command == 'status':
                    self.show_status()
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_interactive_help(self):
        """Show help for interactive mode."""
        help_text = """
Available Commands:
- help: Show this help message
- plan: Generate a plan for a goal
- analyze: Show supervisor analysis
- status: Show current status
- quit/exit: Exit interactive mode
        """
        print(help_text)
    
    def show_status(self):
        """Show current status of all agents."""
        status = {
            'planner_initialized': self.planner is not None,
            'executor_initialized': self.executor is not None,
            'supervisor_initialized': self.supervisor is not None,
            'episode_id': self.episode_id,
            'goal': self.goal
        }
        print(f"Status: {status}")


def main():
    """Main entry point for the agent orchestrator."""
    parser = argparse.ArgumentParser(description="Agent Orchestrator - Run complete agent system")
    parser.add_argument("--goal", type=str, help="Goal to achieve")
    parser.add_argument("--task_file", type=str, help="Path to JSON task file")
    parser.add_argument("--visual_trace_dir", type=str, default="visual_traces", 
                       help="Directory for visual traces")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--env", type=str, help="Environment type (optional)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.goal and not args.task_file and not args.interactive:
        print("❌ Error: Must provide --goal, --task_file, or --interactive")
        parser.print_help()
        return
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        goal=args.goal,
        task_file=args.task_file,
        visual_trace_dir=args.visual_trace_dir
    )
    
    # Run based on mode
    if args.interactive:
        orchestrator.run_interactive_mode()
    else:
        # Load task from file if provided
        if args.task_file:
            task_config = orchestrator.load_task_from_file(args.task_file)
            if task_config:
                orchestrator.task_config = task_config  # Store the full task config
                orchestrator.goal = task_config.get('goal', args.goal)
        
        if not orchestrator.goal:
            print("❌ Error: No goal specified")
            return
        
        # Run complete workflow
        results = orchestrator.run_complete_workflow()
        
        # Display summary
        if 'error' not in results:
            print("\n📋 WORKFLOW SUMMARY:")
            print(f"Episode ID: {results.get('episode_id')}")
            print(f"Goal: {results.get('goal')}")
            print(f"Subtasks: {results.get('planning', {}).get('subtasks_generated', 0)}")
            
            # Show evaluation grade if available
            evaluation = results.get('evaluation', {})
            if evaluation and 'overall_assessment' in evaluation:
                assessment = evaluation['overall_assessment']
                print(f"Overall Grade: {assessment.get('grade', 'N/A')}")
                print(f"Overall Score: {assessment.get('overall_score', 0):.2f}")
        else:
            print(f"❌ Workflow failed: {results.get('error')}")


if __name__ == "__main__":
    main() 