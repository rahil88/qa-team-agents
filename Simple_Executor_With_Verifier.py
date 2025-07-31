#!/usr/bin/env python3
"""
Simple Executor with Verifier Integration
Calls verifier agent after every subtask execution
"""

import os
import json
import time
from typing import Dict, Any, List
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
from android_world.env import env_launcher
from Verifier_Agent import VerifierAgent
from Supervisor_Agent import SupervisorAgent

# Set up the Android environment and grounding agent
env = env_launcher.load_and_setup_env(
    console_port=5554,
    emulator_setup=False,
    freeze_datetime=True,
    adb_path="/Users/craddy-san/Library/Android/sdk/platform-tools/adb",
    grpc_port=8554,
)

# Use the grounding agent for ALL UI interactions
grounding_agent = AndroidEnvGroundingAgent(env)

class SimpleExecutorWithVerifier:
    """
    Simple executor that calls verifier after each subtask
    """
    def __init__(self, grounding_agent, subtasks, episode_id: str = None):
        self.agent = grounding_agent
        self.subtasks = subtasks
        self.verifier = VerifierAgent(env, subtasks)
        self.execution_results = []
        
        # Initialize supervisor agent for visual trace recording
        self.supervisor = SupervisorAgent(visual_trace_dir="visual_traces")
        self.episode_id = episode_id or f"episode_{int(time.time())}"
        
    def execute_subtasks_with_verification(self, goal: str = "Execute subtasks with verification"):
        """Execute subtasks and verify after each one"""
        print("🚀 STARTING EXECUTION WITH VERIFICATION")
        print("=" * 50)
        
        # Start supervisor episode recording
        self.supervisor.start_episode_recording(self.episode_id, goal)
        
        for i, subtask in enumerate(self.subtasks):
            print(f"\n🎯 SUBTASK {i+1}/{len(self.subtasks)}: {subtask['name']}")
            print("-" * 40)
            
            # Record frame before execution using supervisor
            self._record_frame_before_execution(subtask)
            
            # Execute the subtask
            subtask_result = self._execute_single_subtask(subtask)
            self.execution_results.append(subtask_result)
            
            # Record frame after execution using supervisor
            self._record_frame_after_execution(subtask, subtask_result)
            
            # Verify after execution
            verification_result = self._verify_subtask(subtask, subtask_result)
            
            # Display results
            self._display_subtask_results(subtask_result, verification_result)
            
            # Brief pause between subtasks
            time.sleep(1)
        
        # Final comprehensive verification
        print("\n🔍 FINAL COMPREHENSIVE VERIFICATION")
        print("=" * 50)
        final_verification = self._final_verification()
        
        # End supervisor episode recording
        episode_summary = self.supervisor.end_episode_recording("completed")
        print(f"🎬 Supervisor visual trace recording completed: {episode_summary['trace_directory']}")
        
        return {
            'subtask_results': self.execution_results,
            'final_verification': final_verification,
            'visual_trace_summary': episode_summary
        }
    
    def _record_frame_before_execution(self, subtask):
        """Record a frame before executing a subtask using supervisor"""
        try:
            # Get current UI state
            state = env.get_state(wait_to_stabilize=True)
            ui_state = self._extract_ui_summary(state)
            
            # Add subtask information to UI state
            ui_state['subtask_info'] = {
                'name': subtask['name'],
                'phase': 'before_execution'
            }
            
            # Record frame using supervisor
            frame_metadata = self.supervisor.record_step(
                env=env,
                ui_state=ui_state,
                agent_action=f"before_{subtask['name']}"
            )
            
            if frame_metadata:
                print(f"📸 Supervisor recorded frame before execution: {frame_metadata['filename']}")
                
        except Exception as e:
            print(f"⚠️ Error recording frame before execution: {e}")
    
    def _record_frame_after_execution(self, subtask, subtask_result):
        """Record a frame after executing a subtask using supervisor"""
        try:
            # Get current UI state
            state = env.get_state(wait_to_stabilize=True)
            ui_state = self._extract_ui_summary(state)
            
            # Add execution result to UI state
            ui_state['execution_result'] = {
                'success': subtask_result.get('execution_success', False),
                'error': subtask_result.get('error_message'),
                'action_executed': subtask_result.get('action_executed')
            }
            
            # Record frame using supervisor
            frame_metadata = self.supervisor.record_step(
                env=env,
                ui_state=ui_state,
                agent_action=f"after_{subtask['name']}"
            )
            
            if frame_metadata:
                print(f"📸 Supervisor recorded frame after execution: {frame_metadata['filename']}")
                
        except Exception as e:
            print(f"⚠️ Error recording frame after execution: {e}")
    
    def _execute_single_subtask(self, subtask):
        """Execute a single subtask - one simple action based on subtask name"""
        subtask_name = subtask["name"].lower()
        
        print(f"📱 Processing: {subtask_name}")
        
        # Convert the subtask name directly to a single action
        action = self._convert_subtask_name_to_action(subtask_name)
        
        result = {
            'subtask_name': subtask['name'],
            'action_executed': action,
            'execution_success': False,
            'error_message': None,
            'ui_state_before': None,
            'ui_state_after': None
        }
        
        if action:
            print(f"🔄 Action: {action}")
            
            # Capture UI state before
            try:
                state_before = env.get_state(wait_to_stabilize=True)
                result['ui_state_before'] = self._extract_ui_summary(state_before)
            except Exception as e:
                print(f"⚠️ Could not capture UI state before: {e}")
            
            # Execute action
            try:
                obs, reward, done, info = self.agent.step(action)
                result['execution_success'] = True
                
                # Improved wait timing based on action type
                if "wait" in subtask_name:
                    time.sleep(3)  # Wait longer for explicit wait commands
                elif "swipe" in subtask_name:
                    time.sleep(1)  # Short wait for swipe actions to complete
                elif "open" in subtask_name and "app" in subtask_name:
                    time.sleep(6)  # Longer wait for app launches
                    # Force UI state refresh for app launches
                    print("🔄 Forcing UI state refresh after app launch...")
                    for i in range(2):
                        time.sleep(1)
                        state = env.get_state(wait_to_stabilize=True)
                        print(f"   UI refresh {i+1}/2 completed")
                else:
                    time.sleep(2)  # Normal wait for other actions
                
            except Exception as e:
                result['error_message'] = str(e)
                print(f"❌ Execution failed: {e}")
            
            # Capture UI state after
            try:
                state_after = env.get_state(wait_to_stabilize=True)
                result['ui_state_after'] = self._extract_ui_summary(state_after)
            except Exception as e:
                print(f"⚠️ Could not capture UI state after: {e}")
            
        else:
            result['error_message'] = "Could not convert subtask to action"
            print("⚠️ Could not convert subtask to action")
        
        return result
    
    def _verify_subtask(self, subtask, subtask_result):
        """Verify the subtask execution"""
        print("🔍 Verifying subtask execution...")
        
        try:
            # Create a single-subtask list for verification
            single_subtask = [subtask]
            temp_verifier = VerifierAgent(env, single_subtask)
            
            # Run verification
            results, final_verdict = temp_verifier.verify(enable_adaptive_mode=False)
            
            return {
                'verification_success': True,
                'subtask_result': results[0] if results else None,
                'final_verdict': final_verdict
            }
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return {
                'verification_success': False,
                'error': str(e),
                'subtask_result': None,
                'final_verdict': None
            }
    
    def _final_verification(self):
        """Perform final comprehensive verification"""
        print("🔍 Running final comprehensive verification...")
        
        try:
            results, final_verdict = self.verifier.verify(enable_adaptive_mode=True)
            
            return {
                'verification_success': True,
                'all_results': results,
                'final_verdict': final_verdict
            }
            
        except Exception as e:
            print(f"❌ Final verification failed: {e}")
            return {
                'verification_success': False,
                'error': str(e),
                'all_results': [],
                'final_verdict': None
            }
    
    def _display_subtask_results(self, subtask_result, verification_result):
        """Display results for a subtask"""
        print(f"\n📊 SUBTASK RESULTS:")
        print(f"   Execution: {'✅ SUCCESS' if subtask_result['execution_success'] else '❌ FAILED'}")
        
        if verification_result['verification_success']:
            subtask_verification = verification_result['subtask_result']
            if subtask_verification:
                result = subtask_verification.get('result', 'unknown')
                print(f"   Verification: {'✅ PASS' if result == 'pass' else '❌ FAIL'}")
                
                # Show any bugs or problems detected
                bugs = subtask_verification.get('bugs_detected', [])
                problems = subtask_verification.get('problems_detected', [])
                
                if bugs:
                    print(f"   🐛 Bugs detected: {len(bugs)}")
                if problems:
                    print(f"   ⚠️ Problems detected: {len(problems)}")
        else:
            print(f"   Verification: ❌ FAILED - {verification_result.get('error', 'Unknown error')}")
    
    def _convert_subtask_name_to_action(self, subtask_name):
        """Convert subtask name to action using proper grounding agent format"""
        
        # Settings app actions
        if "open settings app" in subtask_name or "tap settings" in subtask_name:
            return self._create_tap_action("settings")
            
        elif "tap network internet" in subtask_name or "navigate to wifi settings" in subtask_name:
            # Try multiple possible texts for network settings
            return self._create_flexible_tap_action(["network & internet", "network", "connections", "connectivity"])
            
        elif "tap internet" in subtask_name or "tap internet option" in subtask_name:
            # Try multiple possible texts for internet settings  
            return self._create_flexible_tap_action(["internet", "mobile network", "data", "network"])
            
        elif "toggle wifi switch" in subtask_name or "enable wifi" in subtask_name or "disable wifi" in subtask_name:
            # Look for the actual Wi-Fi toggle switch in the current screen
            return self._create_flexible_tap_action(["wi-fi", "wifi", "wireless", "toggle", "switch"])
            
        elif "turn on wi-fi" in subtask_name or "turn on wifi" in subtask_name or "tap wi-fi" in subtask_name:
            return self._create_flexible_tap_action(["wi-fi", "wifi", "wireless", "internet"])
            
        elif "turn off wi-fi" in subtask_name or "turn off wifi" in subtask_name:
            return self._create_flexible_tap_action(["wi-fi", "wifi", "wireless", "internet"])
            
        elif "swipe down twice" in subtask_name:
            # For Pixel 6, swipe down twice to open quick settings
            return {"action_type": "swipe", "direction": "down"}
            
        elif "swipe down from top" in subtask_name:
            # Swipe down from top to open notification panel
            return {"action_type": "swipe", "direction": "down"}
            
        elif "swipe down again" in subtask_name:
            # Second swipe down to expand quick settings
            return {"action_type": "swipe", "direction": "down"}
            
        elif "tap wifi tile" in subtask_name or "tap wifi quick settings tile" in subtask_name:
            # Enhanced Wi-Fi tile detection for quick settings
            return self._create_wifi_tile_action()
            
        elif "long press wifi tile" in subtask_name:
            return self._create_tap_action("wi-fi")  # Can be enhanced for long press
            
        elif "tap back" in subtask_name:
            return {"action_type": "key", "key": "back"}
            
        elif "swipe up" in subtask_name:
            return {"action_type": "swipe", "direction": "up"}
            
        elif "wait" in subtask_name:
            return {"action_type": "wait"}
        
        # Generic fallback using the old method
        return self._convert_step_to_action(subtask_name)
    
    def _extract_ui_summary(self, state):
        """Extract a summary of the UI state"""
        try:
            ui_elements = getattr(state, 'ui_elements', [])
            if ui_elements:
                # Convert to dict format if needed
                if not isinstance(ui_elements[0], dict):
                    ui_elements = self._convert_ui_elements_to_dict(ui_elements)
                
                # Get first few text elements as summary
                text_elements = [elem.get('text', '') for elem in ui_elements[:5] if elem.get('text')]
                
                return {
                    'ui_elements': ui_elements,
                    'ui_tree': str(state),
                    'screen_title': getattr(state, 'screen_title', ''),
                    'current_activity': getattr(state, 'current_activity', ''),
                    'summary': f"UI with {len(ui_elements)} elements, texts: {text_elements[:3]}"
                }
            else:
                return {
                    'ui_elements': [],
                    'ui_tree': str(state),
                    'screen_title': getattr(state, 'screen_title', ''),
                    'current_activity': getattr(state, 'current_activity', ''),
                    'summary': "No UI elements"
                }
        except Exception as e:
            return {
                'ui_elements': [],
                'ui_tree': str(state),
                'screen_title': getattr(state, 'screen_title', ''),
                'current_activity': getattr(state, 'current_activity', ''),
                'summary': f"UI state error: {e}"
            }
    
    def _convert_ui_elements_to_dict(self, ui_elements):
        """Convert UI element objects to dictionary format"""
        converted_elements = []
        for el in ui_elements:
            element_dict = {
                'text': getattr(el, 'text', None),
                'content_description': getattr(el, 'content_description', None), 
                'class_name': getattr(el, 'class_name', None),
                'is_clickable': getattr(el, 'is_clickable', None),
                'is_enabled': getattr(el, 'is_enabled', None),
                'bounds': None
            }
            
            # Handle bounding box - try both bbox and bbox_pixels
            bbox = getattr(el, 'bbox', None) or getattr(el, 'bbox_pixels', None)
            if bbox:
                element_dict['bounds'] = [
                    getattr(bbox, 'x_min', 0),
                    getattr(bbox, 'y_min', 0), 
                    getattr(bbox, 'x_max', 0),
                    getattr(bbox, 'y_max', 0)
                ]
            
            converted_elements.append(element_dict)
        return converted_elements
    
    def _create_tap_action(self, target_text):
        """Create a tap action for finding and clicking an element with specific text"""
        # Get current UI state with stabilization to ensure fresh elements
        state = env.get_state(wait_to_stabilize=True)
        
        # State object has ui_elements attribute directly, no need to convert
        ui_elements = getattr(state, 'ui_elements', [])
        
        # Convert ui_elements to dict format if needed
        if ui_elements and not isinstance(ui_elements[0], dict):
            ui_elements = self._convert_ui_elements_to_dict(ui_elements)
        
        # Find element with matching text
        target = self._find_element_by_text(ui_elements, target_text)
        
        if target:
            print(f"🎯 Found target element: {target}")
            
            # Try to get bounds from different possible attributes
            bounds = None
            if target.get('bounds'):
                bounds = target['bounds']
            elif hasattr(target, 'bounds'):
                bounds = target.bounds
            elif target.get('position'):
                bounds = target['position']
            
            if bounds:
                # Calculate center coordinates
                if len(bounds) >= 4:
                    x = (bounds[0] + bounds[2]) / 2
                    y = (bounds[1] + bounds[3]) / 2
                elif len(bounds) == 2:
                    x, y = bounds[0], bounds[1]
                else:
                    print(f"⚠️ Unexpected bounds format: {bounds}")
                    x, y = 540, 1200  # Default center
                
                # Get actual screen size from environment
                try:
                    screen_width, screen_height = env.logical_screen_size
                    print(f"📱 Screen size: {screen_width}x{screen_height}")
                except:
                    # Fallback to common Pixel 6 resolution
                    screen_width = 1080
                    screen_height = 2400
                    print(f"📱 Using fallback screen size: {screen_width}x{screen_height}")
                
                # Calculate normalized coordinates
                norm_x = x / screen_width
                norm_y = y / screen_height
                
                print(f"🎯 Element bounds: {bounds}")
                print(f"🎯 Center coordinates: ({int(x)}, {int(y)})")
                print(f"🎯 Normalized coordinates: ({norm_x:.3f}, {norm_y:.3f})")
                
                return {
                    "action_type": "touch",
                    "touch_position": [norm_x, norm_y]
                }
            else:
                print(f"⚠️ No bounds found for target element: {target}")
        
        # Fallback: try to click by approximate location
        print(f"⚠️ Using fallback center tap for: {target_text}")
        return {"action_type": "touch", "touch_position": [0.5, 0.5]}
    
    def _create_flexible_tap_action(self, target_texts):
        """Create a tap action that tries multiple possible text matches"""
        for target_text in target_texts:
            action = self._create_tap_action(target_text)
            if action.get("touch_position") != [0.5, 0.5]:  # Not fallback
                return action
        return self._create_tap_action(target_texts[0])  # Use first as fallback
    
    def _create_wifi_tile_action(self):
        """Create action specifically for Wi-Fi tile in quick settings"""
        state = env.get_state(wait_to_stabilize=True)
        ui_elements = getattr(state, 'ui_elements', [])
        
        # Convert ui_elements to dict format if needed
        if ui_elements and not isinstance(ui_elements[0], dict):
            ui_elements = self._convert_ui_elements_to_dict(ui_elements)
        
        print("🔍 Looking for Wi-Fi tile in quick settings...")
        print(f"📊 Total UI elements found: {len(ui_elements)}")
        
        # Updated Wi-Fi tile search patterns for Pixel 6
        # The Wi-Fi toggle appears as "Internet" tile in quick settings
        wifi_patterns = [
            'internet', 'Internet', 'INTERNET',  # Primary - this is how Wi-Fi appears on Pixel 6
            'wi-fi', 'wifi', 'wireless', 'Wi-Fi', 'WiFi', 'Wireless',  # Fallback patterns
            'network', 'Network', 't-mobile', 'T-Mobile', '5g', '5G'  # Network indicators
        ]
        
        # First try exact matches for Internet/Wi-Fi related text
        print(f"🔍 Searching for patterns: {wifi_patterns}")
        for pattern in wifi_patterns:
            for element in ui_elements:
                text = element.get('text') or ''
                desc = element.get('content_description') or ''
                
                if (pattern in text or pattern in desc) and element.get('is_clickable', False):
                    print(f"✅ Found Wi-Fi tile match: '{text}' / '{desc}'")
                    return self._create_tap_action(text)
        
        # Fallback to generic tap action
        return self._create_tap_action("internet")
    
    def _find_element_by_text(self, ui_elements, target_text):
        """Find UI element by text content (case-insensitive, partial match)"""
        target_text = target_text.lower()
        
        print(f"🔍 Looking for: '{target_text}'")
        print(f"📋 Available UI elements:")
        
        # Debug: print all available elements
        for i, element in enumerate(ui_elements):
            text = element.get('text') or ''
            desc = element.get('content_description') or ''
            clickable = element.get('is_clickable', False)
            print(f"  [{i}] text='{text}' desc='{desc}' clickable={clickable}")
        
        # Try exact matches first
        for element in ui_elements:
            if element.get('text') and target_text == element['text'].lower():
                print(f"✅ Found exact match: '{element['text']}'")
                return element
                
        # Try partial matches with multiple keywords
        target_keywords = target_text.split()
        for element in ui_elements:
            # Handle None values properly
            text = element.get('text') or ''
            desc = element.get('content_description') or ''
            element_text = (text + ' ' + desc).lower()
            
            # Check if all keywords are present
            if all(keyword in element_text for keyword in target_keywords):
                print(f"✅ Found partial match: '{element.get('text', '')}'")
                return element
        
        # Try individual keyword matches for common terms
        common_mappings = {
            'network & internet': ['network', 'internet', 'connections', 'wireless', 'network & internet', 'connected devices'],
            'internet': ['internet', 'network', 'wifi', 'wi-fi', 'mobile data', 'mobile network'],
            'wi-fi': ['wi-fi', 'wifi', 'wireless', 'internet', 'network'],
            'settings': ['settings', 'setting']
        }
        
        if target_text in common_mappings:
            for keyword in common_mappings[target_text]:
                for element in ui_elements:
                    text = element.get('text') or ''
                    desc = element.get('content_description') or ''
                    if keyword.lower() in text.lower() or keyword.lower() in desc.lower():
                        print(f"✅ Found mapped match: '{element.get('text', '')}'")
                        return element
        
        print(f"❌ No match found for: '{target_text}'")
        return None
    
    def _convert_step_to_action(self, step):
        """Convert a step description to an action"""
        step_lower = step.lower()
        
        if "tap" in step_lower or "click" in step_lower:
            # Extract the target from the step
            if "settings" in step_lower:
                return self._create_tap_action("settings")
            elif "wifi" in step_lower or "wi-fi" in step_lower:
                return self._create_tap_action("wifi")
            elif "network" in step_lower:
                return self._create_tap_action("network")
            else:
                return self._create_smart_tap_action(step)
        elif "swipe" in step_lower:
            if "down" in step_lower:
                return {"action_type": "swipe", "direction": "down"}
            elif "up" in step_lower:
                return {"action_type": "swipe", "direction": "up"}
            else:
                return {"action_type": "swipe", "direction": "down"}
        elif "wait" in step_lower:
            return {"action_type": "wait"}
        else:
            return self._create_smart_tap_action(step)
    
    def _create_smart_tap_action(self, step):
        """Smart action creation that tries to find any mentioned UI element"""
        # Extract potential UI element names from the step
        keywords = ["settings", "wi-fi", "wifi", "network", "internet", "toggle", "switch", "button"]
        
        for keyword in keywords:
            if keyword in step.lower():
                return self._create_tap_action(keyword)
                
        # Default fallback
        return {"action_type": "wait"}

def main():
    """Main entry point"""
    # Load subtasks
    try:
        with open("subtasks.json", "r") as f:
            subtasks = json.load(f)
    except FileNotFoundError:
        print("❌ subtasks.json not found")
        return
    
    # Create executor with verifier
    executor = SimpleExecutorWithVerifier(grounding_agent, subtasks)
    
    # Execute with verification
    result = executor.execute_subtasks_with_verification()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"simple_executor_verifier_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    # Print final summary
    print("\n🎯 FINAL SUMMARY:")
    total_subtasks = len(result['subtask_results'])
    successful_executions = sum(1 for r in result['subtask_results'] if r['execution_success'])
    print(f"   Executions: {successful_executions}/{total_subtasks} successful")
    
    if result['final_verification']['verification_success']:
        final_verdict = result['final_verification']['final_verdict']
        if final_verdict:
            print(f"   Final Verdict: {final_verdict.get('verdict', 'UNKNOWN')}")
    
    return result

if __name__ == "__main__":
    main() 