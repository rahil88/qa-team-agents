import json
import os
import logging
from datetime import datetime
from android_world.env import env_launcher
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
from typing import Dict, Any, List, Optional

# 1. Load subtasks (planner goal)
with open("subtasks.json", "r") as f:
    subtasks = json.load(f)

# 2. Set up Android environment and grounding agent (reuse config)
env = env_launcher.load_and_setup_env(
    console_port=5554,
    emulator_setup=False,
    freeze_datetime=True,
    adb_path="/Users/craddy-san/Library/Android/sdk/platform-tools/adb",
    grpc_port=8554,
)
grounding_agent = AndroidEnvGroundingAgent(env)

class VerifierAgent:
    """
    Enhanced Verifier Agent that:
    - Receives Planner Goal, Executor Result, and UI State  
    - Determines if current state matches expectation (pass/fail)
    - Detects functional bugs (missing screen, wrong toggle state)
    - Uses heuristics + reasoning over UI hierarchy
    - Provides adaptive recommendations for mid-execution replanning
    - Logs all interactions and provides detailed verdicts
    """
    
    def __init__(self, env, subtasks, executor_result=None, log_file=None):
        self.env = env
        self.subtasks = subtasks  # Planner Goal
        self.executor_result = executor_result  # Executor Result
        
        # Setup comprehensive logging
        self.log_file = log_file or f"verifier_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.interaction_log = []
        self.agent_decisions = {}
        self.verification_start_time = datetime.now()
        
        # Extract goal from subtasks
        self.expected_goal = self._determine_expected_goal(subtasks)
        print(f"🎯 Expected Goal: {self.expected_goal}")
        
        # Log initialization
        self._log_interaction("verifier_initialized", {
            "expected_goal": self.expected_goal,
            "subtasks_count": len(subtasks),
            "subtasks": [task['name'] for task in subtasks],
            "timestamp": self.verification_start_time.isoformat()
        })

    def _log_interaction(self, interaction_type: str, data: Dict[str, Any]):
        """Log an interaction with timestamp and metadata"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "data": data
        }
        self.interaction_log.append(interaction)
        
        # Also print to console for immediate feedback
        print(f"📝 LOG: {interaction_type} - {data.get('summary', 'No summary')}")

    def _log_agent_decision(self, agent_name: str, decision: str, reasoning: str, confidence: float = 1.0):
        """Log a decision made by a specific agent"""
        if agent_name not in self.agent_decisions:
            self.agent_decisions[agent_name] = []
            
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence
        }
        self.agent_decisions[agent_name].append(decision_record)
        
        print(f"🤖 {agent_name.upper()}: {decision} (confidence: {confidence:.2f})")
        print(f"   Reasoning: {reasoning}")

    def verify(self, enable_adaptive_mode=True):
        """Main verification method with comprehensive logging"""
        print("\n🔍 STARTING ENHANCED VERIFICATION")
        print("=" * 50)
        
        # Log verification start
        self._log_interaction("verification_started", {
            "enable_adaptive_mode": enable_adaptive_mode,
            "expected_goal": self.expected_goal
        })
        
        # Get current UI state
        state = self.env.get_state()
        ui_elements = state.ui_elements
        
        # Log UI state capture
        self._log_interaction("ui_state_captured", {
            "ui_elements_count": len(ui_elements),
            "screen_size": getattr(self.env, 'logical_screen_size', 'unknown')
        })
        
        # Analyze current state
        ui_analysis = self._analyze_ui_state(ui_elements)
        current_screen = ui_analysis['screen_type']
        functional_state = ui_analysis['functional_state']
        
        # Log screen analysis
        self._log_interaction("screen_analysis_completed", {
            "detected_screen": current_screen,
            "confidence": ui_analysis['screen_confidence'],
            "indicators_found": ui_analysis['screen_indicators'],
            "functional_state": functional_state
        })
        
        print(f"📱 Current Screen: {current_screen}")
        print(f"🔧 Functional State: {functional_state}")
        
        # Determine if state matches expectation
        verification_result = self._verify_goal_achievement(self.expected_goal, ui_analysis)
        
        # Log goal verification decision
        self._log_agent_decision("goal_verifier", 
                                "PASS" if verification_result[0] else "FAIL",
                                verification_result[1],
                                ui_analysis['screen_confidence'])
        
        # Detect functional bugs
        bugs_detected = self._detect_functional_bugs(ui_analysis)
        
        # Log bug detection
        if bugs_detected:
            self._log_interaction("bugs_detected", {
                "bug_count": len(bugs_detected),
                "bugs": bugs_detected
            })
            self._log_agent_decision("bug_detector", 
                                    f"DETECTED_{len(bugs_detected)}_BUGS",
                                    f"Found {len(bugs_detected)} functional bugs",
                                    0.8 if bugs_detected else 0.2)
        else:
            self._log_agent_decision("bug_detector", 
                                    "NO_BUGS",
                                    "No functional bugs detected",
                                    0.9)
        
        # NEW: Detect problems requiring adaptation
        problems_detected = self._detect_adaptation_problems(ui_analysis)
        
        # Log problem detection
        if problems_detected:
            self._log_interaction("adaptation_problems_detected", {
                "problem_count": len(problems_detected),
                "problems": problems_detected
            })
            self._log_agent_decision("problem_detector", 
                                    f"DETECTED_{len(problems_detected)}_PROBLEMS",
                                    f"Found {len(problems_detected)} problems requiring adaptation",
                                    0.8)
        else:
            self._log_agent_decision("problem_detector", 
                                    "NO_PROBLEMS",
                                    "No adaptation problems detected",
                                    0.9)
        
        # NEW: Generate adaptive recommendations if problems found
        adaptive_recommendations = []
        if enable_adaptive_mode and problems_detected:
            adaptive_recommendations = self._generate_adaptive_recommendations(problems_detected, ui_analysis)
            
            # Log adaptive recommendations
            self._log_interaction("adaptive_recommendations_generated", {
                "recommendation_count": len(adaptive_recommendations),
                "recommendations": adaptive_recommendations
            })
            self._log_agent_decision("adaptive_planner", 
                                    f"GENERATED_{len(adaptive_recommendations)}_RECOMMENDATIONS",
                                    f"Generated {len(adaptive_recommendations)} adaptive recommendations",
                                    0.7)
        
        # Create detailed results for each subtask
        results = []
        overall_success = verification_result[0]
        
        # Make analysis JSON-safe by removing non-serializable objects
        json_safe_analysis = {
            'screen_type': ui_analysis['screen_type'],
            'screen_confidence': ui_analysis['screen_confidence'],
            'screen_indicators': ui_analysis['screen_indicators'],
            'functional_state': ui_analysis['functional_state'],
            'total_elements': ui_analysis['total_elements']
        }
        
        # Log per-subtask analysis
        for i, subtask in enumerate(self.subtasks):
            subtask_result = "pass" if overall_success else "fail"
            self._log_interaction("subtask_analyzed", {
                "subtask_index": i,
                "subtask_name": subtask['name'],
                "result": subtask_result,
                "reason": verification_result[1]
            })
            
            result = {
                "subtask": subtask['name'],
                "result": subtask_result, 
                "reason": verification_result[1],
                "ui_state": current_screen,
                "functional_state": functional_state,
                "bugs_detected": bugs_detected,
                "detailed_analysis": json_safe_analysis,
                # NEW: Add adaptive capabilities
                "problems_detected": problems_detected,
                "adaptive_recommendations": adaptive_recommendations,
                "requires_replanning": len(problems_detected) > 0
            }
            results.append(result)
        
        # Generate final verdict
        final_verdict = self._generate_final_verdict(verification_result, bugs_detected, problems_detected, adaptive_recommendations)
        
        # Log final verdict
        self._log_interaction("final_verdict_generated", final_verdict)
        self._log_agent_decision("verdict_generator", 
                                final_verdict['verdict'],
                                final_verdict['reasoning'],
                                final_verdict['confidence'])
        
        # Summary
        self._print_verification_summary(verification_result, bugs_detected, problems_detected, adaptive_recommendations, final_verdict)
        
        # Save complete log to file
        self._save_complete_log(results, final_verdict)
        
        return results, final_verdict

    def _generate_final_verdict(self, verification_result, bugs_detected, problems_detected, adaptive_recommendations) -> Dict[str, Any]:
        """Generate comprehensive final verdict with confidence scoring"""
        
        # Base verdict from goal achievement
        goal_achieved = verification_result[0]
        goal_reason = verification_result[1]
        
        # Calculate confidence based on multiple factors
        confidence_factors = []
        
        # Factor 1: Goal achievement (primary)
        if goal_achieved:
            confidence_factors.append(("goal_achieved", 0.9))
        else:
            confidence_factors.append(("goal_not_achieved", 0.1))
        
        # Factor 2: Bug presence (negative impact)
        if bugs_detected:
            bug_severity = max(bug.get('severity', 'low') for bug in bugs_detected)
            severity_weights = {"high": 0.3, "medium": 0.5, "low": 0.7}
            confidence_factors.append(("bugs_present", severity_weights.get(bug_severity, 0.5)))
        else:
            confidence_factors.append(("no_bugs", 0.9))
        
        # Factor 3: Problem presence (negative impact)
        if problems_detected:
            problem_severity = max(problem.get('severity', 'low') for problem in problems_detected)
            severity_weights = {"high": 0.4, "medium": 0.6, "low": 0.8}
            confidence_factors.append(("problems_present", severity_weights.get(problem_severity, 0.6)))
        else:
            confidence_factors.append(("no_problems", 0.9))
        
        # Factor 4: Adaptive recommendations available (positive for recovery)
        if adaptive_recommendations:
            confidence_factors.append(("adaptation_possible", 0.7))
        
        # Calculate weighted confidence
        total_confidence = sum(weight for _, weight in confidence_factors) / len(confidence_factors)
        
        # Determine verdict category
        if goal_achieved and not bugs_detected:
            verdict = "PASSED"
            reasoning = f"Goal successfully achieved: {goal_reason}"
        elif goal_achieved and bugs_detected:
            verdict = "PASSED_WITH_BUGS"
            reasoning = f"Goal achieved but {len(bugs_detected)} bugs detected: {goal_reason}"
        elif not goal_achieved and adaptive_recommendations:
            verdict = "FAILED_BUT_RECOVERABLE"
            reasoning = f"Goal not achieved but {len(adaptive_recommendations)} recovery options available: {goal_reason}"
        else:
            verdict = "FAILED"
            reasoning = f"Goal not achieved and no recovery options: {goal_reason}"
        
        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "confidence": total_confidence,
            "goal_achieved": goal_achieved,
            "bugs_count": len(bugs_detected),
            "problems_count": len(problems_detected),
            "adaptation_available": len(adaptive_recommendations) > 0,
            "confidence_factors": confidence_factors,
            "timestamp": datetime.now().isoformat()
        }

    def _save_complete_log(self, results, final_verdict):
        """Save complete verification log to JSON file"""
        complete_log = {
            "verification_session": {
                "start_time": self.verification_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "expected_goal": self.expected_goal,
                "final_verdict": final_verdict
            },
            "interaction_log": self.interaction_log,
            "agent_decisions": self.agent_decisions,
            "subtask_results": results,
            "summary": {
                "total_interactions": len(self.interaction_log),
                "total_decisions": sum(len(decisions) for decisions in self.agent_decisions.values()),
                "agents_involved": list(self.agent_decisions.keys()),
                "verdict": final_verdict['verdict'],
                "confidence": final_verdict['confidence']
            }
        }
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(complete_log, f, indent=2)
            print(f"📄 Complete log saved to: {self.log_file}")
        except Exception as e:
            print(f"⚠️ Failed to save log file: {e}")
    
    def _determine_expected_goal(self, subtasks):
        """Determine what the final goal should be based on subtasks"""
        task_names = [task['name'].lower() for task in subtasks]
        
        # Analyze subtask patterns to determine goal
        if any("wifi" in name or "wi-fi" in name for name in task_names):
            # Check if this is quick settings approach or settings app approach
            if any("swipe down" in name for name in task_names) and any("tile" in name for name in task_names):
                return "wifi_quick_toggle"  # Quick settings approach
            elif any("settings" in name for name in task_names):
                return "wifi_settings_access"  # Settings app approach
            else:
                return "wifi_control"
        elif any("chrome" in name for name in task_names):
            return "browser_access"
        elif any("swipe" in name for name in task_names):
            return "notification_panel_access"
        else:
            return "unknown_goal"
    
    def _analyze_ui_state(self, ui_elements) -> Dict[str, Any]:
        """Analyze current UI state with enhanced heuristics"""
        
        # Convert ui_elements if needed
        if ui_elements and not isinstance(ui_elements[0], dict):
            ui_elements = self._convert_ui_elements_to_dict(ui_elements)
        
        # Screen type detection
        screen_indicators = {
            'notification_panel': ['brightness', 'bluetooth', 'airplane', 'internet', 'quick settings'],
            'quick_settings': ['wifi', 'bluetooth', 'airplane', 'brightness', 'internet', 'network'],
            'settings_app': ['network & internet', 'connected devices', 'apps', 'notifications'],
            'home_screen': ['phone', 'messages', 'chrome', 'gmail', 'camera'],
            'wifi_settings': ['saved networks', 'network preferences', 'wi-fi', 'connected']
        }
        
        # Count indicators for each screen type
        element_texts = []
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            element_texts.extend([text, desc])
        
        all_text = ' '.join(element_texts)
        
        screen_scores = {}
        for screen_type, indicators in screen_indicators.items():
            score = sum(1 for indicator in indicators if indicator in all_text)
            if score > 0:
                screen_scores[screen_type] = score / len(indicators)
        
        if screen_scores:
            detected_screen = max(screen_scores.keys(), key=lambda x: screen_scores[x])
            confidence = screen_scores[detected_screen]
            matching_indicators = [ind for ind in screen_indicators[detected_screen] if ind in all_text]
        else:
            detected_screen = 'unknown'
            confidence = 0.0
            matching_indicators = []
        
        print(f"🎯 Screen Detection: {detected_screen} (confidence: {confidence:.2f})")
        print(f"📋 Matching indicators: {matching_indicators}")
        
        # Enhanced functional state analysis
        functional_state = {
            'wifi_toggle_available': False,
            'wifi_state': 'unknown',
            'bluetooth_toggle_available': False,
            'bluetooth_state': 'unknown',
            'navigation_options': [],
            'error_indicators': []
        }
        
        # Look for Wi-Fi/Internet toggle - Updated to recognize Internet tile
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            
            # Check for Internet tile (which is the Wi-Fi toggle on Pixel 6)
            if ('internet' in desc or 'internet' in text) and element.get('is_clickable', False):
                functional_state['wifi_toggle_available'] = True
                # Check the state from the text
                element_text = element.get('text', '').lower()
                if 'on' in element_text:
                    functional_state['wifi_state'] = 'on'
                elif 'off' in element_text:
                    functional_state['wifi_state'] = 'off'
                print(f"✅ Found Internet/Wi-Fi toggle: text='{element.get('text', '')}' desc='{element.get('content_description', '')}' state={functional_state['wifi_state']}")
                break
            
            # Fallback: look for traditional wi-fi patterns
            elif any(pattern in text or pattern in desc for pattern in ['wi-fi', 'wifi', 'wireless']):
                if element.get('is_clickable', False):
                    functional_state['wifi_toggle_available'] = True
                    element_text = element.get('text', '').lower()
                    if 'on' in element_text or 'connected' in element_text:
                        functional_state['wifi_state'] = 'on'
                    elif 'off' in element_text or 'disconnected' in element_text:
                        functional_state['wifi_state'] = 'off'
                    break
        
        # Look for Bluetooth toggle
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            
            if 'bluetooth' in text or 'bluetooth' in desc:
                if element.get('is_clickable', False):
                    functional_state['bluetooth_toggle_available'] = True
                    element_text = element.get('text', '').lower()
                    if 'on' in element_text:
                        functional_state['bluetooth_state'] = 'on'
                    elif 'off' in element_text:
                        functional_state['bluetooth_state'] = 'off'
                    break
        
        # Collect navigation options
        for element in ui_elements:
            if element.get('is_clickable', False):
                desc = element.get('content_description', '').lower()
                if desc and desc not in ['', 'clickable']:
                    functional_state['navigation_options'].append(desc)
        
        # Look for error indicators
        error_patterns = ['error', 'failed', 'unable', 'cannot', 'problem']
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            if any(pattern in text or pattern in desc for pattern in error_patterns):
                functional_state['error_indicators'].append(f"{text} {desc}".strip())
        
        return {
            'screen_type': detected_screen,
            'screen_confidence': confidence,
            'screen_indicators': matching_indicators,
            'functional_state': functional_state,
            'total_elements': len(ui_elements)
        }
    
    def _convert_ui_elements_to_dict(self, ui_elements):
        """Converts a list of AndroidElement objects to a list of dictionaries."""
        return [
            {
                'text': element.text,
                'content_description': element.content_description,
                'is_clickable': element.is_clickable,
                'element': element
            }
            for element in ui_elements
        ]
    
    def _detect_screen_type_enhanced(self, all_texts):
        """Enhanced screen detection with confidence scoring"""
        
        # Define screen patterns with weights
        screen_patterns = {
            'wifi_settings': {
                'indicators': ['wi-fi', 'wifi', 'network preferences', 'saved networks', 'add network', 'wifi settings'],
                'min_matches': 2,
                'weight': 1.0
            },
            'notification_panel': {
                'indicators': ['quick settings', 'clear all', 'notification', 'brightness', 'bluetooth', 'airplane', 'flashlight'],
                'min_matches': 2, 
                'weight': 0.9
            },
            'network_settings': {
                'indicators': ['network & internet', 'internet', 'mobile network', 'data usage', 'hotspot'],
                'min_matches': 1,
                'weight': 0.8
            },
            'settings_main': {
                'indicators': ['network & internet', 'connected devices', 'apps', 'notifications', 'battery'],
                'min_matches': 2,
                'weight': 0.7
            },
            'home_screen': {
                'indicators': ['phone', 'messages', 'chrome', 'gmail', 'camera'],
                'min_matches': 3,
                'weight': 0.6
            }
        }
        
        best_match = {'type': 'unknown_screen', 'confidence': 0.0, 'indicators': []}
        
        for screen_type, pattern in screen_patterns.items():
            matches = []
            match_count = 0
            
            for indicator in pattern['indicators']:
                for text, desc in all_texts:
                    if indicator in text or indicator in desc:
                        matches.append(indicator)
                        match_count += 1
                        break
            
            if match_count >= pattern['min_matches']:
                confidence = (match_count / len(pattern['indicators'])) * pattern['weight']
                
                if confidence > best_match['confidence']:
                    best_match = {
                        'type': screen_type,
                        'confidence': confidence,
                        'indicators': matches
                    }
        
        print(f"🎯 Screen Detection: {best_match['type']} (confidence: {best_match['confidence']:.2f})")
        print(f"📋 Matching indicators: {best_match['indicators']}")
        
        return best_match
    
    def _analyze_functional_state(self, all_texts, clickable_elements):
        """Analyze functional state (toggles, accessibility, etc.)"""
        
        functional_state = {
            'wifi_toggle_available': False,
            'wifi_state': 'unknown',
            'bluetooth_toggle_available': False, 
            'bluetooth_state': 'unknown',
            'navigation_options': [],
            'error_indicators': []
        }
        
        # Check for Wi-Fi state indicators
        for text, desc in all_texts:
            # Wi-Fi state detection
            if any(term in text or term in desc for term in ['wi-fi', 'wifi', 'wireless']):
                functional_state['wifi_toggle_available'] = True
                
                # Try to determine state
                if any(state in text or state in desc for state in ['on', 'enabled', 'connected']):
                    functional_state['wifi_state'] = 'enabled'
                elif any(state in text or state in desc for state in ['off', 'disabled', 'disconnected']):
                    functional_state['wifi_state'] = 'disabled'
            
            # Error indicators
            if any(error in text or error in desc for error in ['error', 'failed', 'not found', 'unavailable']):
                functional_state['error_indicators'].append(text or desc)
        
        # Check navigation options
        for element in clickable_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            
            if any(nav in text or nav in desc for nav in ['settings', 'network', 'internet', 'back', 'home']):
                functional_state['navigation_options'].append(text or desc)
        
        return functional_state
    
    def _verify_goal_achievement(self, expected_goal: str, ui_analysis: Dict[str, Any]) -> tuple:
        """Verify if the expected goal has been achieved based on UI analysis"""
        
        functional_state = ui_analysis['functional_state']
        screen_type = ui_analysis['screen_type']
        
        if expected_goal == "wifi_quick_toggle":
            # SUCCESS: We found the Wi-Fi toggle (Internet tile) and can determine its state
            if functional_state['wifi_toggle_available']:
                wifi_state = functional_state['wifi_state']
                if wifi_state in ['on', 'off']:
                    return True, f"Wi-Fi toggle successfully accessed and is currently {wifi_state}"
                else:
                    return True, f"Wi-Fi toggle found but state unclear: {wifi_state}"
            else:
                return False, f"Wi-Fi toggle not found in {screen_type}"
                
        elif expected_goal == "wifi_settings_access":
            if screen_type == "wifi_settings":
                return True, "Successfully accessed Wi-Fi settings screen"
            elif functional_state['wifi_toggle_available']:
                return True, "Wi-Fi controls accessible"
            else:
                return False, f"Wi-Fi settings not accessible from {screen_type}"
                
        elif expected_goal == "bluetooth_toggle":
            if functional_state['bluetooth_toggle_available']:
                return True, f"Bluetooth toggle found, state: {functional_state['bluetooth_state']}"
            else:
                return False, f"Bluetooth toggle not found in {screen_type}"
                
        elif expected_goal == "notification_access":
            if screen_type in ["notification_panel", "quick_settings"]:
                return True, f"Successfully accessed {screen_type}"
            else:
                return False, f"Expected notification panel but found {screen_type}"
                
        else:
            # Generic success if we can identify the screen and have some functional controls
            if screen_type != "unknown" and (
                functional_state['wifi_toggle_available'] or 
                functional_state['bluetooth_toggle_available'] or 
                len(functional_state['navigation_options']) > 0
            ):
                return True, f"Goal partially achieved - accessible controls found in {screen_type}"
            else:
                return False, f"Goal not achieved - limited functionality in {screen_type}"
    
    def _detect_functional_bugs(self, ui_analysis):
        """Detect functional bugs using heuristics"""
        
        bugs = []
        screen_type = ui_analysis['screen_type']
        functional_state = ui_analysis['functional_state']
        
        # Bug 1: Missing expected UI elements
        if self.expected_goal == "wifi_quick_toggle" and screen_type == "notification_panel":
            if not functional_state['wifi_toggle_available']:
                bugs.append({
                    'type': 'missing_ui_element',
                    'description': 'Wi-Fi toggle not found in notification panel',
                    'severity': 'high'
                })
        
        # Bug 2: Wrong screen navigation
        if self.expected_goal == "wifi_settings_access" and screen_type not in ["wifi_settings", "network_settings"]:
            bugs.append({
                'type': 'navigation_failure', 
                'description': f'Navigation led to {screen_type} instead of Wi-Fi settings',
                'severity': 'high'
            })
        
        # Bug 3: Error indicators present
        if functional_state['error_indicators']:
            bugs.append({
                'type': 'error_state',
                'description': f'Error indicators detected: {functional_state["error_indicators"]}',
                'severity': 'medium'
            })
        
        # Bug 4: Low confidence screen detection
        if ui_analysis['screen_confidence'] < 0.5:
            bugs.append({
                'type': 'ambiguous_ui_state',
                'description': f'Low confidence ({ui_analysis["screen_confidence"]:.2f}) in screen detection',
                'severity': 'low'
            })
        
        return bugs
    
    def _detect_adaptation_problems(self, ui_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect problems that require mid-execution adaptation"""
        problems = []
        ui_elements = getattr(self.env.get_state(), 'ui_elements', [])
        
        # Convert ui_elements if needed
        if ui_elements and not isinstance(ui_elements[0], dict):
            ui_elements = self._convert_ui_elements_to_dict(ui_elements)
        
        # Problem 1: Pop-up dialogs blocking execution
        popup_indicators = ['ok', 'cancel', 'dismiss', 'allow', 'deny', 'close', 'got it', 'continue']
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            
            if element.get('is_clickable', False) and any(indicator in text for indicator in popup_indicators):
                problems.append({
                    'type': 'blocking_popup',
                    'severity': 'high',
                    'description': f'Pop-up dialog detected with button: "{element.get("text", "")}"',
                    'blocking_element': element,
                    'suggested_action': 'dismiss_popup'
                })
        
        # Problem 2: Permission dialogs
        permission_patterns = ['permission', 'access', 'location', 'camera', 'microphone', 'storage']
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            
            if any(pattern in text or pattern in desc for pattern in permission_patterns):
                if any(button in text for button in ['allow', 'deny', 'grant']):
                    problems.append({
                        'type': 'permission_request',
                        'severity': 'high', 
                        'description': f'Permission request detected: "{element.get("text", "")}"',
                        'blocking_element': element,
                        'suggested_action': 'handle_permission'
                    })
        
        # Problem 3: Error states blocking progress
        error_patterns = ['error', 'failed', 'unable to', 'connection problem', 'try again']
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            
            if any(pattern in text or pattern in desc for pattern in error_patterns):
                problems.append({
                    'type': 'error_state',
                    'severity': 'medium',
                    'description': f'Error state detected: "{element.get("text", "")}"',
                    'blocking_element': element,
                    'suggested_action': 'retry_or_alternative'
                })
        
        # Problem 4: Unexpected screen state
        screen_type = ui_analysis['screen_type']
        if self.expected_goal == "wifi_quick_toggle" and screen_type == "home_screen":
            problems.append({
                'type': 'wrong_screen_state',
                'severity': 'medium',
                'description': f'Expected notification panel but found {screen_type}',
                'suggested_action': 'retry_navigation'
            })
        
        # Problem 5: Missing UI elements after navigation
        if self.expected_goal == "wifi_quick_toggle" and screen_type in ["notification_panel", "quick_settings"]:
            if not ui_analysis['functional_state']['wifi_toggle_available']:
                problems.append({
                    'type': 'missing_target_element',
                    'severity': 'high',
                    'description': 'Wi-Fi toggle not found in quick settings',
                    'suggested_action': 'alternative_approach'
                })
        
        # Problem 6: Network connectivity issues
        connectivity_indicators = ['no internet', 'offline', 'network unavailable']
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            
            if any(indicator in text or indicator in desc for indicator in connectivity_indicators):
                problems.append({
                    'type': 'connectivity_issue',
                    'severity': 'low',
                    'description': f'Network issue detected: "{element.get("text", "")}"',
                    'suggested_action': 'acknowledge_and_continue'
                })
        
        return problems

    def _generate_adaptive_recommendations(self, problems: List[Dict[str, Any]], ui_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific adaptive recommendations for detected problems"""
        recommendations = []
        
        for problem in problems:
            problem_type = problem['type']
            
            if problem_type == 'blocking_popup':
                # Recommend dismissing the popup
                element = problem['blocking_element']
                recommendations.append({
                    'action_type': 'dismiss_popup',
                    'priority': 'immediate',
                    'description': 'Dismiss blocking popup dialog',
                    'specific_action': {
                        'action_type': 'touch',
                        'touch_position': self._calculate_element_center(element),
                        'target_text': element.get('text', 'OK')
                    },
                    'fallback_actions': [
                        {'action_type': 'key', 'key': 'back'},
                        {'action_type': 'touch', 'touch_position': [0.5, 0.8]}  # Generic dismiss area
                    ]
                })
            
            elif problem_type == 'permission_request':
                element = problem['blocking_element']
                text = element.get('text', '').lower()
                
                # Choose appropriate response based on text
                if 'allow' in text:
                    action_desc = 'Grant permission to continue'
                elif 'deny' in text:
                    action_desc = 'Deny permission and find alternative'
                else:
                    action_desc = 'Handle permission request'
                
                recommendations.append({
                    'action_type': 'handle_permission',
                    'priority': 'immediate',
                    'description': action_desc,
                    'specific_action': {
                        'action_type': 'touch',
                        'touch_position': self._calculate_element_center(element),
                        'target_text': element.get('text', '')
                    }
                })
            
            elif problem_type == 'wrong_screen_state':
                if self.expected_goal == "wifi_quick_toggle":
                    recommendations.append({
                        'action_type': 'retry_navigation',
                        'priority': 'high',
                        'description': 'Retry opening notification panel',
                        'specific_action': {
                            'action_type': 'swipe',
                            'direction': 'down',
                            'start_position': [0.5, 0.1]  # Start from very top
                        },
                        'alternative_actions': [
                            {
                                'action_type': 'swipe',
                                'direction': 'down',
                                'description': 'Second swipe to expand quick settings'
                            },
                            {
                                'action_type': 'navigate_to_settings',
                                'description': 'Alternative: Use Settings app approach'
                            }
                        ]
                    })
            
            elif problem_type == 'missing_target_element':
                recommendations.append({
                    'action_type': 'alternative_approach',
                    'priority': 'high', 
                    'description': 'Switch to Settings app approach for Wi-Fi',
                    'specific_action': {
                        'action_type': 'touch',
                        'target_text': 'Settings',
                        'description': 'Open Settings app'
                    },
                    'follow_up_steps': [
                        {'action': 'tap', 'target': 'Network & Internet'},
                        {'action': 'tap', 'target': 'Wi-Fi'}
                    ]
                })
            
            elif problem_type == 'error_state':
                recommendations.append({
                    'action_type': 'retry_or_alternative',
                    'priority': 'medium',
                    'description': 'Handle error state',
                    'specific_action': {
                        'action_type': 'wait',
                        'duration': 2,
                        'description': 'Wait for error to clear'
                    },
                    'retry_action': {
                        'action_type': 'retry_last_action',
                        'description': 'Retry the failed action'
                    }
                })
            
            elif problem_type == 'connectivity_issue':
                recommendations.append({
                    'action_type': 'acknowledge_and_continue',
                    'priority': 'low',
                    'description': 'Acknowledge connectivity issue and continue',
                    'specific_action': {
                        'action_type': 'wait',
                        'duration': 1,
                        'description': 'Continue despite connectivity issue'
                    }
                })
        
        return recommendations

    def _calculate_element_center(self, element: Dict[str, Any]) -> List[float]:
        """Calculate normalized center coordinates for an element"""
        bounds = element.get('bounds')
        if bounds and len(bounds) >= 4:
            x = (bounds[0] + bounds[2]) / 2
            y = (bounds[1] + bounds[3]) / 2
            
            # Get screen size for normalization
            try:
                screen_width, screen_height = self.env.logical_screen_size
            except:
                screen_width, screen_height = 1080, 2400
            
            return [x / screen_width, y / screen_height]
        else:
            return [0.5, 0.5]  # Fallback to center

    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """Public method for Planner to get adaptive recommendations"""
        state = self.env.get_state()
        ui_analysis = self._analyze_ui_state(state.ui_elements)
        problems = self._detect_adaptation_problems(ui_analysis)
        recommendations = self._generate_adaptive_recommendations(problems, ui_analysis)
        
        return {
            'has_problems': len(problems) > 0,
            'problems': problems,
            'recommendations': recommendations,
            'priority_level': 'immediate' if any(p['severity'] == 'high' for p in problems) else 'normal'
        }
    
    def _print_verification_summary(self, verification_result, bugs_detected, problems_detected=None, adaptive_recommendations=None, final_verdict=None):
        """Print detailed verification summary"""
        
        print(f"\n📊 VERIFICATION SUMMARY")
        print("=" * 50)
        print(f"🎯 Goal Achievement: {'✅ SUCCESS' if verification_result[0] else '❌ FAILED'}")
        print(f"📝 Reason: {verification_result[1]}")
        
        if final_verdict:
            print(f"🏁 FINAL VERDICT: {final_verdict['verdict']} (confidence: {final_verdict['confidence']:.2f})")
            print(f"   Reasoning: {final_verdict['reasoning']}")
            print(f"   Confidence Factors: {final_verdict['confidence_factors']}")
        
        if bugs_detected:
            print(f"\n🐛 BUGS DETECTED ({len(bugs_detected)}):")
            for i, bug in enumerate(bugs_detected):
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[bug['severity']]
                print(f"  {i+1}. {severity_icon} {bug['type']}: {bug['description']}")
        else:
            print(f"\n✅ NO FUNCTIONAL BUGS DETECTED")

        if problems_detected:
            print(f"\n🔧 PROBLEMS DETECTED ({len(problems_detected)}):")
            for i, problem in enumerate(problems_detected):
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[problem['severity']]
                print(f"  {i+1}. {severity_icon} {problem['type']}: {problem['description']}")
        elif problems_detected is not None:
            print(f"\n✅ NO PROBLEMS DETECTED")

        if adaptive_recommendations:
            print(f"\n🔧 ADAPTIVE RECOMMENDATIONS ({len(adaptive_recommendations)}):")
            for i, rec in enumerate(adaptive_recommendations):
                print(f"  {i+1}. {rec['priority']}: {rec['description']}")
                if 'specific_action' in rec:
                    action = rec['specific_action']
                    print(f"    Action: {action['action_type']}")
                    if 'touch_position' in action:
                        print(f"    Position: {action['touch_position']}")
                    if 'target_text' in action:
                        print(f"    Target Text: {action['target_text']}")
                    if 'fallback_actions' in rec:
                        print(f"    Fallback Actions:")
                        for fa in rec['fallback_actions']:
                            print(f"      - {fa['action_type']}")
                    if 'follow_up_steps' in rec:
                        print(f"    Follow-up Steps:")
                        for step in rec['follow_up_steps']:
                            print(f"      - {step['action']} to {step['target']}")
                if 'retry_action' in rec:
                    print(f"    Retry Action: {rec['retry_action']['action_type']}")
        elif adaptive_recommendations is not None:
            print(f"\n✅ NO ADAPTIVE RECOMMENDATIONS")

        if final_verdict:
            print(f"\n🏁 FINAL VERDICT: {final_verdict['verdict']} (confidence: {final_verdict['confidence']:.2f})")
            print(f"   Reasoning: {final_verdict['reasoning']}")
            print(f"   Confidence Factors: {final_verdict['confidence_factors']}")

# Standalone execution for testing
if __name__ == "__main__":
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
    
    # Run verification
    verifier = VerifierAgent(env, subtasks)
    results, final_verdict = verifier.verify()
    
    print("\n🎉 VERIFICATION RESULTS:")
    print(json.dumps(results, indent=2))
    print("\n🏁 FINAL VERDICT:")
    print(json.dumps(final_verdict, indent=2))
