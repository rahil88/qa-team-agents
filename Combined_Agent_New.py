import os
import json
import time
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
from android_world.env import env_launcher

# Import the existing Executor and Verifier agents
from Executor_Agent import SmartExecutor
from Executor_Agent import grounding_agent
from Executor_Agent import env
from Verifier_Agent import VerifierAgent

class SimpleCombinedAgent:
    """
    Simple orchestrator that calls existing Executor and Verifier agents
    """
    
    def __init__(self):
        # Use the same environment and grounding agent from Executor_Agent
        self.env = env
        self.grounding_agent = grounding_agent
        
    def run_combined_test(self):
        """Run executor followed by verifier using existing agent implementations"""
        
        # 1. Load subtasks
        with open("subtasks.json", "r") as f:
            subtasks = json.load(f)
        
        print("🚀 STARTING SIMPLE COMBINED EXECUTOR + VERIFIER TEST")
        print("=" * 60)
        print(f"📋 Loaded {len(subtasks)} subtasks from subtasks.json")
        
        # Display subtasks
        for i, subtask in enumerate(subtasks):
            print(f"  {i+1}. {subtask['name']}")
        
        # 2. Run Executor using existing SmartExecutor
        print("\n🎯 PHASE 1: EXECUTING SUBTASKS")
        print("-" * 40)
        executor = SmartExecutor(self.grounding_agent)
        
        try:
            executor.execute_subtasks(subtasks)
            print("✅ Executor completed all subtasks")
            executor_success = True
        except Exception as e:
            print(f"❌ Executor failed with error: {e}")
            executor_success = False
        
        print("\n⏳ Waiting for UI to stabilize...")
        time.sleep(1)  # Reduced from 3 to 1 second to prevent notification panel auto-close
        
        # 4. Run Verifier using existing VerifierAgent
        print("\n🔍 PHASE 2: VERIFYING COMPLETION")
        print("-" * 40)
        
        try:
            verifier = VerifierAgent(self.env, subtasks)
            results, final_verdict = verifier.verify()
            print("✅ Verifier completed analysis")
            
            # Display final verdict
            print(f"\n🏁 FINAL VERDICT: {final_verdict['verdict']} (confidence: {final_verdict['confidence']:.2f})")
            print(f"   Reasoning: {final_verdict['reasoning']}")
            
        except Exception as e:
            print(f"❌ Verifier failed with error: {e}")
            # Create fallback results
            results = []
            for subtask in subtasks:
                results.append({
                    "subtask": subtask['name'],
                    "result": "fail" if not executor_success else "unknown",
                    "reason": f"Verifier error: {e}"
                })
        
        # 5. Display final results
        print("\n🎉 FINAL RESULTS:")
        print("=" * 60)
        print(json.dumps(results, indent=2))
        
        # 6. Summary
        passed = sum(1 for r in results if r["result"] == "pass")
        total = len(results)
        print(f"\n📊 SUMMARY: {passed}/{total} subtasks passed")
        
        if passed == total:
            print("✅ ALL SUBTASKS COMPLETED SUCCESSFULLY!")
        elif passed > 0:
            print("⚠️ Some subtasks passed, some failed")
        else:
            print("❌ ALL SUBTASKS FAILED")
        
        return results

def main():
    """Main entry point"""
    agent = SimpleCombinedAgent()
    agent.run_combined_test()

if __name__ == "__main__":
    main() 