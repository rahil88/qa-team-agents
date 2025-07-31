#!/usr/bin/env python3
"""
Test script for Enhanced Adaptive Verifier Agent
Demonstrates mid-execution adaptation capabilities
"""

import json
import time
from android_world.env import env_launcher
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
from Verifier_Agent import VerifierAgent
from Adaptive_Interface import AdaptiveInterface

def test_adaptive_verifier():
    """Test the adaptive capabilities of the enhanced Verifier Agent"""
    
    print("🚀 TESTING ADAPTIVE VERIFIER AGENT")
    print("=" * 60)
    
    # Load subtasks
    with open("subtasks.json", "r") as f:
        subtasks = json.load(f)
    
    # Set up environment
    env = env_launcher.load_and_setup_env(
        console_port=5554,
        emulator_setup=False,
        freeze_datetime=True,
        adb_path="/Users/craddy-san/Library/Android/sdk/platform-tools/adb",
        grpc_port=8554,
    )
    
    grounding_agent = AndroidEnvGroundingAgent(env)
    
    # Create enhanced verifier with adaptive mode
    verifier = VerifierAgent(env, subtasks)
    adaptive_interface = AdaptiveInterface(verifier)
    
    print(f"\n📋 Loaded {len(subtasks)} subtasks:")
    for i, task in enumerate(subtasks):
        print(f"  {i+1}. {task['name']}")
    
    # Test 1: Check for problems before starting execution
    print(f"\n🔍 TEST 1: Initial Problem Detection")
    print("-" * 40)
    
    initial_problems = adaptive_interface.check_for_problems()
    print(f"Has problems: {initial_problems['has_problems']}")
    print(f"Priority level: {initial_problems.get('priority_level', 'none')}")
    
    if initial_problems['has_problems']:
        print(f"Found {len(initial_problems['problems'])} problems:")
        for problem in initial_problems['problems']:
            print(f"  - {problem['type']}: {problem['description']} (severity: {problem['severity']})")
    
    # Test 2: Simulate execution and check for adaptation needs
    print(f"\n🔍 TEST 2: Simulated Execution with Adaptation")
    print("-" * 40)
    
    # Simulate first subtask execution (swipe down)
    print("\n📱 Executing: swipe down from top")
    action = {"action_type": "swipe", "direction": "down"}
    obs, reward, done, info = grounding_agent.step(action)
    time.sleep(2)
    
    # Check if adaptation is needed
    should_adapt = adaptive_interface.should_adapt("medium")
    print(f"Should adapt: {should_adapt}")
    
    if should_adapt:
        print("\n🔧 ADAPTATION REQUIRED!")
        
        # Get immediate actions
        immediate_actions = adaptive_interface.get_immediate_actions()
        if immediate_actions:
            print(f"Immediate actions needed: {len(immediate_actions)}")
            for action in immediate_actions:
                print(f"  Action: {action['action_type']}")
        
        # Get alternative strategies
        alternatives = adaptive_interface.get_alternative_strategies()
        if alternatives:
            print(f"Alternative strategies available: {len(alternatives)}")
            for alt in alternatives:
                print(f"  Strategy: {alt['strategy']} - {alt['description']}")
    
    # Test 3: Full verification with adaptive mode
    print(f"\n🔍 TEST 3: Full Verification with Adaptive Mode")
    print("-" * 40)
    
    results = verifier.verify(enable_adaptive_mode=True)
    
    # Test 4: Problem summary
    print(f"\n🔍 TEST 4: Problem Summary Analysis")
    print("-" * 40)
    
    summary = adaptive_interface.get_problem_summary()
    print(f"Total problems: {summary['total_problems']}")
    print(f"Severity breakdown: {summary['severity_breakdown']}")
    print(f"Problem types: {summary['problem_types']}")
    print(f"Requires immediate action: {summary['requires_immediate_action']}")
    print(f"Has alternatives: {summary['has_alternatives']}")
    
    # Test 5: Display adaptive recommendations
    print(f"\n🔍 TEST 5: Adaptive Recommendations Detail")
    print("-" * 40)
    
    for i, result in enumerate(results):
        if result.get('adaptive_recommendations'):
            print(f"\nSubtask {i+1}: {result['subtask']}")
            print(f"Requires replanning: {result['requires_replanning']}")
            
            for j, rec in enumerate(result['adaptive_recommendations']):
                print(f"  Recommendation {j+1}:")
                print(f"    Type: {rec['action_type']}")
                print(f"    Priority: {rec['priority']}")
                print(f"    Description: {rec['description']}")
                
                if 'specific_action' in rec:
                    specific = rec['specific_action']
                    print(f"    Specific Action: {specific['action_type']}")
                    if 'touch_position' in specific:
                        print(f"    Position: {specific['touch_position']}")
    
    print(f"\n🎉 ADAPTIVE VERIFIER TEST COMPLETED")
    print("=" * 60)
    
    return results

def test_planner_integration_example():
    """Example of how a Planner would integrate with adaptive capabilities"""
    
    print("\n🔗 PLANNER INTEGRATION EXAMPLE")
    print("=" * 50)
    
    # Simulated planner workflow with adaptive capabilities
    subtasks = [
        {"name": "swipe down from top", "info": "Open notification panel"},
        {"name": "tap wifi tile", "info": "Toggle wifi state"}
    ]
    
    # Mock environment setup (in real code, use actual environment)
    print("📋 Planner receives subtasks...")
    print("🔄 Executor begins execution...")
    
    # Simulated execution loop with adaptation
    for i, subtask in enumerate(subtasks):
        print(f"\n📱 Executing subtask {i+1}: {subtask['name']}")
        
        # Simulate execution
        time.sleep(1)
        
        # Check for problems (this would use real adaptive interface)
        print("🔍 Verifier checking for problems...")
        
        # Simulate different problem scenarios
        if i == 0:
            # Simulate popup blocking execution
            print("⚠️ Problem detected: Blocking popup found")
            print("🔧 Adaptive recommendation: Dismiss popup before continuing")
            print("✅ Executing adaptive action: Tap 'OK' button")
            
        elif i == 1:
            # Simulate missing UI element
            print("⚠️ Problem detected: Wi-Fi tile not found in notification panel")
            print("🔧 Adaptive recommendation: Switch to Settings app approach")
            print("📱 Replanning: Using Settings > Network & Internet > Wi-Fi")
    
    print("\n✅ Planner successfully adapted to problems during execution!")

if __name__ == "__main__":
    # Run the adaptive verifier test
    test_results = test_adaptive_verifier()
    
    # Show planner integration example
    test_planner_integration_example()
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print("=" * 40)
    print(f"Subtasks tested: {len(test_results)}")
    
    adaptation_count = sum(1 for result in test_results if result.get('requires_replanning', False))
    print(f"Subtasks requiring adaptation: {adaptation_count}")
    
    recommendation_count = sum(len(result.get('adaptive_recommendations', [])) for result in test_results)
    print(f"Total adaptive recommendations: {recommendation_count}") 