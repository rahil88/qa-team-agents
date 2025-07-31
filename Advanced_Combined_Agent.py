#!/usr/bin/env python3
"""
Advanced Combined Agent with Adaptive Execution and Verification
Integrates Executor and Verifier with mid-execution replanning capabilities
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
from android_world.env import env_launcher

# Import the existing Executor and Verifier agents
from Executor_Agent import SmartExecutor, grounding_agent, env
from Verifier_Agent import VerifierAgent
from Adaptive_Interface import AdaptiveInterface

class AdvancedCombinedAgent:
    """
    Advanced orchestrator that integrates Executor and Verifier with adaptive capabilities
    """
    
    def __init__(self, enable_adaptive_mode: bool = True, max_retries: int = 3):
        # Use the same environment and grounding agent from Executor_Agent
        self.env = env
        self.grounding_agent = grounding_agent
        self.enable_adaptive_mode = enable_adaptive_mode
        self.max_retries = max_retries
        
        # Initialize components
        self.executor = SmartExecutor(self.grounding_agent)
        self.verifier = None  # Will be initialized with subtasks
        self.adaptive_interface = None  # Will be initialized with verifier
        
        # Execution state
        self.execution_history = []
        self.current_subtask_index = 0
        self.retry_count = 0
        self.adaptive_recommendations_used = []
        
    def run_advanced_test(self, subtasks_file: str = "subtasks.json"):
        """Run advanced executor + verifier with adaptive capabilities"""
        
        # 1. Load and validate subtasks
        subtasks = self._load_subtasks(subtasks_file)
        
        print("🚀 STARTING ADVANCED COMBINED EXECUTOR + VERIFIER TEST")
        print("=" * 70)
        print(f"📋 Loaded {len(subtasks)} subtasks from {subtasks_file}")
        print(f"🔧 Adaptive Mode: {'✅ ENABLED' if self.enable_adaptive_mode else '❌ DISABLED'}")
        print(f"🔄 Max Retries: {self.max_retries}")
        
        # Display subtasks
        for i, subtask in enumerate(subtasks):
            print(f"  {i+1}. {subtask['name']}")
        
        # 2. Initialize verifier and adaptive interface
        self.verifier = VerifierAgent(self.env, subtasks)
        if self.enable_adaptive_mode:
            self.adaptive_interface = AdaptiveInterface(self.verifier)
        
        # 3. Execute with adaptive capabilities
        print("\n🎯 PHASE 1: ADAPTIVE EXECUTION")
        print("-" * 50)
        
        execution_result = self._execute_with_adaptation(subtasks)
        
        # 4. Final verification and analysis
        print("\n🔍 PHASE 2: COMPREHENSIVE VERIFICATION")
        print("-" * 50)
        
        verification_result = self._comprehensive_verification(subtasks)
        
        # 5. Generate final report
        print("\n📊 PHASE 3: FINAL ANALYSIS & REPORT")
        print("-" * 50)
        
        final_report = self._generate_final_report(execution_result, verification_result)
        
        return final_report
    
    def _load_subtasks(self, subtasks_file: str) -> List[Dict[str, Any]]:
        """Load and validate subtasks from file"""
        try:
            with open(subtasks_file, "r") as f:
                subtasks = json.load(f)
            
            # Validate subtasks structure
            for i, subtask in enumerate(subtasks):
                if 'name' not in subtask:
                    raise ValueError(f"Subtask {i} missing 'name' field")
            
            return subtasks
        except FileNotFoundError:
            print(f"❌ Subtasks file '{subtasks_file}' not found")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in subtasks file: {e}")
            raise
    
    def _execute_with_adaptation(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute subtasks with adaptive capabilities and mid-execution replanning"""
        
        execution_result = {
            'subtasks_executed': [],
            'adaptations_made': [],
            'retries_performed': 0,
            'execution_success': True,
            'final_state': 'unknown'
        }
        
        self.current_subtask_index = 0
        
        while self.current_subtask_index < len(subtasks):
            current_subtask = subtasks[self.current_subtask_index]
            subtask_name = current_subtask['name']
            
            print(f"\n🎯 Executing subtask {self.current_subtask_index + 1}/{len(subtasks)}: {subtask_name}")
            
            # Execute single subtask
            subtask_result = self._execute_single_subtask_with_adaptation(current_subtask)
            execution_result['subtasks_executed'].append(subtask_result)
            
            # Check if we need to adapt
            if self.enable_adaptive_mode and subtask_result['requires_adaptation']:
                adaptation_result = self._handle_adaptation(subtask_result, subtasks)
                execution_result['adaptations_made'].append(adaptation_result)
                
                # Check if adaptation was successful
                if adaptation_result['adaptation_success']:
                    print(f"✅ Adaptation successful: {adaptation_result['adaptation_type']}")
                    # Continue with next subtask
                    self.current_subtask_index += 1
                else:
                    print(f"❌ Adaptation failed: {adaptation_result['reason']}")
                    # Retry current subtask or move to next
                    if self.retry_count < self.max_retries:
                        self.retry_count += 1
                        print(f"🔄 Retrying subtask (attempt {self.retry_count}/{self.max_retries})")
                        continue
                    else:
                        print(f"❌ Max retries reached for subtask: {subtask_name}")
                        execution_result['execution_success'] = False
                        break
            else:
                # No adaptation needed, move to next subtask
                self.current_subtask_index += 1
                self.retry_count = 0  # Reset retry count for next subtask
            
            # Brief pause between subtasks
            time.sleep(0.5)
        
        execution_result['final_state'] = 'completed' if execution_result['execution_success'] else 'failed'
        return execution_result
    
    def _execute_single_subtask_with_adaptation(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask and check for adaptation needs"""
        
        subtask_result = {
            'subtask_name': subtask['name'],
            'execution_success': False,
            'requires_adaptation': False,
            'adaptation_reason': None,
            'execution_time': 0,
            'ui_state_before': None,
            'ui_state_after': None,
            'error_message': None
        }
        
        start_time = time.time()
        
        try:
            # Capture UI state before execution
            state_before = self.env.get_state(wait_to_stabilize=True)
            subtask_result['ui_state_before'] = self._extract_ui_summary(state_before)
            
            # Execute the subtask
            self.executor._execute_single_subtask(subtask)
            
            # Capture UI state after execution
            time.sleep(1)  # Allow UI to stabilize
            state_after = self.env.get_state(wait_to_stabilize=True)
            subtask_result['ui_state_after'] = self._extract_ui_summary(state_after)
            
            # Check if adaptation is needed
            if self.enable_adaptive_mode:
                adaptation_check = self._check_adaptation_needed(subtask_result)
                subtask_result['requires_adaptation'] = adaptation_check['needed']
                subtask_result['adaptation_reason'] = adaptation_check['reason']
            
            subtask_result['execution_success'] = True
            
        except Exception as e:
            subtask_result['error_message'] = str(e)
            subtask_result['requires_adaptation'] = True
            subtask_result['adaptation_reason'] = f"Execution error: {e}"
        
        subtask_result['execution_time'] = time.time() - start_time
        return subtask_result
    
    def _check_adaptation_needed(self, subtask_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check if adaptation is needed based on subtask result"""
        
        # Use the adaptive interface to check for problems
        if self.adaptive_interface:
            problems = self.adaptive_interface.check_for_problems()
            
            if problems['has_problems']:
                # Check if any problems require immediate adaptation
                high_priority_problems = [p for p in problems['problems'] 
                                        if p['severity'] in ['high', 'immediate']]
                
                if high_priority_problems:
                    return {
                        'needed': True,
                        'reason': f"High priority problems detected: {[p['type'] for p in high_priority_problems]}"
                    }
        
        # Check for UI state issues
        if subtask_result['ui_state_before'] and subtask_result['ui_state_after']:
            if subtask_result['ui_state_before'] == subtask_result['ui_state_after']:
                return {
                    'needed': True,
                    'reason': "UI state unchanged after action - possible navigation failure"
                }
        
        return {'needed': False, 'reason': None}
    
    def _handle_adaptation(self, subtask_result: Dict[str, Any], subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle adaptation when problems are detected"""
        
        adaptation_result = {
            'adaptation_success': False,
            'adaptation_type': 'unknown',
            'reason': 'No adaptation strategy available',
            'actions_taken': []
        }
        
        if not self.adaptive_interface:
            adaptation_result['reason'] = "Adaptive interface not available"
            return adaptation_result
        
        try:
            # Get adaptive recommendations
            recommendations = self.adaptive_interface.check_for_problems()
            
            if not recommendations['has_problems']:
                adaptation_result['reason'] = "No problems detected for adaptation"
                return adaptation_result
            
            # Get immediate actions
            immediate_actions = self.adaptive_interface.get_immediate_actions()
            
            if immediate_actions:
                # Execute immediate actions
                for action in immediate_actions:
                    print(f"🔄 Executing adaptive action: {action}")
                    try:
                        self.grounding_agent.step(action)
                        adaptation_result['actions_taken'].append(action)
                        time.sleep(1)  # Wait for action to complete
                    except Exception as e:
                        print(f"❌ Adaptive action failed: {e}")
                
                adaptation_result['adaptation_type'] = 'immediate_actions'
                adaptation_result['adaptation_success'] = True
                adaptation_result['reason'] = f"Executed {len(immediate_actions)} immediate actions"
                
            else:
                # Get alternative strategies
                alternatives = self.adaptive_interface.get_alternative_strategies()
                
                if alternatives:
                    # Use the first alternative strategy
                    strategy = alternatives[0]
                    print(f"🔄 Using alternative strategy: {strategy['strategy']}")
                    
                    # Execute alternative actions
                    for action in strategy['actions']:
                        print(f"🔄 Executing alternative action: {action}")
                        try:
                            self.grounding_agent.step(action)
                            adaptation_result['actions_taken'].append(action)
                            time.sleep(1)
                        except Exception as e:
                            print(f"❌ Alternative action failed: {e}")
                    
                    adaptation_result['adaptation_type'] = 'alternative_strategy'
                    adaptation_result['adaptation_success'] = True
                    adaptation_result['reason'] = f"Used alternative strategy: {strategy['strategy']}"
                else:
                    adaptation_result['reason'] = "No adaptation strategies available"
            
        except Exception as e:
            adaptation_result['reason'] = f"Adaptation failed with error: {e}"
        
        return adaptation_result
    
    def _comprehensive_verification(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive verification of the entire execution"""
        
        print("🔍 Performing comprehensive verification...")
        
        try:
            # Use the verifier to analyze the complete execution
            results, final_verdict = self.verifier.verify()
            
            verification_result = {
                'verification_success': True,
                'subtask_results': results,
                'final_verdict': final_verdict,
                'verification_summary': {
                    'total_subtasks': len(subtasks),
                    'passed_subtasks': sum(1 for r in results if r.get('result') == 'pass'),
                    'failed_subtasks': sum(1 for r in results if r.get('result') == 'fail'),
                    'bugs_detected': sum(len(r.get('bugs_detected', [])) for r in results),
                    'problems_detected': sum(len(r.get('problems_detected', [])) for r in results)
                }
            }
            
        except Exception as e:
            verification_result = {
                'verification_success': False,
                'error': str(e),
                'subtask_results': [],
                'final_verdict': None,
                'verification_summary': {}
            }
        
        return verification_result
    
    def _generate_final_report(self, execution_result: Dict[str, Any], verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        final_report = {
            'execution_summary': {
                'total_subtasks': len(execution_result['subtasks_executed']),
                'successful_subtasks': sum(1 for s in execution_result['subtasks_executed'] if s['execution_success']),
                'failed_subtasks': sum(1 for s in execution_result['subtasks_executed'] if not s['execution_success']),
                'adaptations_made': len(execution_result['adaptations_made']),
                'retries_performed': execution_result['retries_performed'],
                'execution_success': execution_result['execution_success']
            },
            'verification_summary': verification_result.get('verification_summary', {}),
            'final_verdict': verification_result.get('final_verdict', {}),
            'adaptive_analysis': {
                'adaptations_used': execution_result['adaptations_made'],
                'recommendations_followed': self.adaptive_recommendations_used
            },
            'detailed_results': {
                'execution_details': execution_result['subtasks_executed'],
                'verification_details': verification_result.get('subtask_results', [])
            }
        }
        
        # Print summary
        print("\n📊 EXECUTION SUMMARY:")
        print(f"   ✅ Successful subtasks: {final_report['execution_summary']['successful_subtasks']}/{final_report['execution_summary']['total_subtasks']}")
        print(f"   🔄 Adaptations made: {final_report['execution_summary']['adaptations_made']}")
        print(f"   🔁 Retries performed: {final_report['execution_summary']['retries_performed']}")
        
        if verification_result.get('final_verdict'):
            verdict = verification_result['final_verdict']
            print(f"\n🏁 FINAL VERDICT: {verdict.get('verdict', 'UNKNOWN')} (confidence: {verdict.get('confidence', 0):.2f})")
            print(f"   Reasoning: {verdict.get('reasoning', 'No reasoning provided')}")
        
        # Determine overall success
        execution_success = final_report['execution_summary']['execution_success']
        verification_success = verification_result.get('verification_success', False)
        
        if execution_success and verification_success:
            print("\n🎉 OVERALL RESULT: ✅ SUCCESS")
        elif execution_success:
            print("\n⚠️ OVERALL RESULT: PARTIAL SUCCESS (execution succeeded, verification failed)")
        else:
            print("\n❌ OVERALL RESULT: FAILED")
        
        return final_report
    
    def _extract_ui_summary(self, state) -> str:
        """Extract a summary of the UI state"""
        try:
            ui_elements = getattr(state, 'ui_elements', [])
            if ui_elements:
                # Get first few text elements as summary
                text_elements = [elem.get('text', '') for elem in ui_elements[:5] if elem.get('text')]
                return f"UI with {len(ui_elements)} elements, texts: {text_elements[:3]}"
            else:
                return "No UI elements"
        except:
            return "UI state unknown"

def main():
    """Main entry point for advanced combined agent"""
    agent = AdvancedCombinedAgent(enable_adaptive_mode=True, max_retries=3)
    result = agent.run_advanced_test()
    
    # Save detailed report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"advanced_agent_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    return result

if __name__ == "__main__":
    main() 