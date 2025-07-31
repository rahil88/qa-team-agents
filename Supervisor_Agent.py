import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import cv2
from PIL import Image
import io
import base64
import time

class MockLLMClient:
    """
    Mock LLM client for testing when Gemini 2.5 is not available.
    Provides realistic analysis responses for development and testing.
    """
    
    def __init__(self):
        self.analysis_templates = {
            'prompt_improvements': [
                "Consider adding more specific UI element targeting in prompts",
                "Include error handling scenarios in action descriptions",
                "Add timing expectations for UI state changes",
                "Specify fallback actions when primary targets are not found"
            ],
            'failure_patterns': [
                "Navigation failures due to UI element changes",
                "Timing issues with app loading and state stabilization",
                "Insufficient error recovery mechanisms",
                "Missing validation of action success"
            ],
            'coverage_recommendations': [
                "Add edge case testing for network connectivity issues",
                "Include testing for different screen orientations",
                "Test with various app states and configurations",
                "Add performance testing for slow network conditions"
            ]
        }
    
    def analyze_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock analysis of episode data"""
        return {
            'prompt_improvements': self.analysis_templates['prompt_improvements'][:2],
            'failure_patterns': self.analysis_templates['failure_patterns'][:2],
            'coverage_recommendations': self.analysis_templates['coverage_recommendations'][:2],
            'confidence_score': 0.85,
            'analysis_timestamp': datetime.now().isoformat()
        }

class EpisodeAnalyzer:
    """
    Analyzes episode data including visual traces and execution logs.
    Provides intelligent insights using LLM analysis.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or MockLLMClient()
    
    def analyze_episode_trace(self, episode_dir: str, execution_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a complete episode trace including visual frames and execution data.
        
        Args:
            episode_dir: Directory containing episode visual traces
            execution_logs: List of execution logs from the episode
            
        Returns:
            Comprehensive episode analysis
        """
        try:
            # Load episode metadata
            metadata_path = os.path.join(episode_dir, "trace_metadata.json")
            if not os.path.exists(metadata_path):
                return {'error': 'Episode metadata not found'}
            
            with open(metadata_path, 'r') as f:
                episode_metadata = json.load(f)
            
            # Analyze visual trace patterns
            visual_analysis = self._analyze_visual_trace(episode_metadata)
            
            # Analyze execution patterns
            execution_analysis = self._analyze_execution_patterns(execution_logs or [])
            
            # Combine data for LLM analysis
            episode_data = {
                'metadata': episode_metadata,
                'visual_analysis': visual_analysis,
                'execution_analysis': execution_analysis,
                'combined_insights': self._combine_insights(visual_analysis, execution_analysis)
            }
            
            # Get LLM analysis
            llm_analysis = self.llm_client.analyze_episode(episode_data)
            
            return {
                'episode_id': episode_metadata.get('episode_id'),
                'goal': episode_metadata.get('goal'),
                'visual_analysis': visual_analysis,
                'execution_analysis': execution_analysis,
                'llm_analysis': llm_analysis,
                'overall_assessment': self._generate_overall_assessment(visual_analysis, execution_analysis, llm_analysis),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _analyze_visual_trace(self, episode_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual trace patterns and UI state changes."""
        
        frames = episode_metadata.get('frames', [])
        if not frames:
            return {'error': 'No frames found in episode'}
        
        analysis = {
            'total_frames': len(frames),
            'frame_sequence_analysis': self._analyze_frame_sequence(frames),
            'ui_state_changes': self._analyze_ui_state_changes(frames),
            'action_effectiveness': self._analyze_action_effectiveness(frames),
            'visual_anomalies': self._detect_visual_anomalies(frames)
        }
        
        return analysis
    
    def _analyze_frame_sequence(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the sequence of frames for patterns and issues."""
        
        if len(frames) < 2:
            return {'message': 'Insufficient frames for sequence analysis'}
        
        # Analyze frame timing
        frame_timings = []
        for i, frame in enumerate(frames):
            if i > 0:
                prev_time = datetime.fromisoformat(frames[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(frame['timestamp'])
                duration = (curr_time - prev_time).total_seconds()
                frame_timings.append(duration)
        
        # Analyze action patterns
        action_patterns = {}
        for frame in frames:
            action = frame.get('agent_action', 'unknown')
            if action not in action_patterns:
                action_patterns[action] = 0
            action_patterns[action] += 1
        
        return {
            'average_frame_interval': sum(frame_timings) / len(frame_timings) if frame_timings else 0,
            'total_duration': sum(frame_timings),
            'action_distribution': action_patterns,
            'frame_count': len(frames)
        }
    
    def _analyze_ui_state_changes(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze UI state changes across frames."""
        
        ui_changes = []
        screen_transitions = []
        
        for i, frame in enumerate(frames):
            ui_state = frame.get('ui_state', {})
            
            # Track screen changes
            screen_title = ui_state.get('screen_title', '')
            if screen_title and (i == 0 or screen_title != frames[i-1].get('ui_state', {}).get('screen_title', '')):
                screen_transitions.append({
                    'frame': i,
                    'screen': screen_title,
                    'action': frame.get('agent_action', 'unknown')
                })
            
            # Track UI element changes
            ui_elements = ui_state.get('ui_elements', [])
            if ui_elements:
                ui_changes.append({
                    'frame': i,
                    'element_count': len(ui_elements),
                    'clickable_elements': len([e for e in ui_elements if e.get('is_clickable', False)]),
                    'action': frame.get('agent_action', 'unknown')
                })
        
        return {
            'screen_transitions': screen_transitions,
            'ui_element_changes': ui_changes,
            'total_screen_changes': len(screen_transitions),
            'average_ui_elements': sum(change['element_count'] for change in ui_changes) / len(ui_changes) if ui_changes else 0
        }
    
    def _analyze_action_effectiveness(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the effectiveness of actions based on UI state changes."""
        
        action_effectiveness = {}
        
        for i, frame in enumerate(frames):
            if i == 0:
                continue
            
            action = frame.get('agent_action', 'unknown')
            prev_frame = frames[i-1]
            
            # Check if action led to UI changes
            prev_ui = prev_frame.get('ui_state', {})
            curr_ui = frame.get('ui_state', {})
            
            ui_changed = (
                prev_ui.get('screen_title') != curr_ui.get('screen_title') or
                len(prev_ui.get('ui_elements', [])) != len(curr_ui.get('ui_elements', []))
            )
            
            if action not in action_effectiveness:
                action_effectiveness[action] = {'total': 0, 'effective': 0}
            
            action_effectiveness[action]['total'] += 1
            if ui_changed:
                action_effectiveness[action]['effective'] += 1
        
        # Calculate effectiveness rates
        for action, stats in action_effectiveness.items():
            if stats['total'] > 0:
                stats['effectiveness_rate'] = stats['effective'] / stats['total']
            else:
                stats['effectiveness_rate'] = 0
        
        return action_effectiveness
    
    def _detect_visual_anomalies(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect visual anomalies in the frame sequence."""
        
        anomalies = []
        
        for i, frame in enumerate(frames):
            ui_state = frame.get('ui_state', {})
            
            # Check for empty UI states
            ui_elements = ui_state.get('ui_elements', [])
            if len(ui_elements) < 5:
                anomalies.append({
                    'frame': i,
                    'type': 'sparse_ui',
                    'description': f'Frame {i} has only {len(ui_elements)} UI elements',
                    'severity': 'medium'
                })
            
            # Check for execution errors
            execution_result = ui_state.get('execution_result', {})
            if execution_result.get('error'):
                anomalies.append({
                    'frame': i,
                    'type': 'execution_error',
                    'description': f'Execution error: {execution_result.get("error")}',
                    'severity': 'high'
                })
            
            # Check for long delays
            if i > 0:
                prev_time = datetime.fromisoformat(frames[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(frame['timestamp'])
                duration = (curr_time - prev_time).total_seconds()
                if duration > 10:  # More than 10 seconds between frames
                    anomalies.append({
                        'frame': i,
                        'type': 'long_delay',
                        'description': f'Long delay of {duration:.1f} seconds between frames',
                        'severity': 'medium'
                    })
        
        return anomalies
    
    def _analyze_execution_patterns(self, execution_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution patterns from logs."""
        
        if not execution_logs:
            return {'message': 'No execution logs provided'}
        
        analysis = {
            'total_actions': len(execution_logs),
            'successful_actions': sum(1 for log in execution_logs if log.get('execution_success', False)),
            'failed_actions': sum(1 for log in execution_logs if not log.get('execution_success', True)),
            'action_types': {},
            'error_patterns': [],
            'performance_metrics': self._calculate_performance_metrics(execution_logs)
        }
        
        # Analyze action types
        for log in execution_logs:
            action = log.get('action_executed', {})
            if isinstance(action, dict):
                action_type = action.get('action_type', 'unknown')
            else:
                action_type = str(action)
            
            if action_type not in analysis['action_types']:
                analysis['action_types'][action_type] = {'total': 0, 'successful': 0}
            
            analysis['action_types'][action_type]['total'] += 1
            if log.get('execution_success', False):
                analysis['action_types'][action_type]['successful'] += 1
        
        # Calculate success rates
        for action_type, stats in analysis['action_types'].items():
            if stats['total'] > 0:
                stats['success_rate'] = stats['successful'] / stats['total']
            else:
                stats['success_rate'] = 0
        
        # Analyze error patterns
        for log in execution_logs:
            if log.get('error_message'):
                analysis['error_patterns'].append({
                    'action': log.get('action_executed'),
                    'error': log.get('error_message'),
                    'subtask': log.get('subtask_name')
                })
        
        return analysis
    
    def _calculate_performance_metrics(self, execution_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from execution logs."""
        
        if not execution_logs:
            return {}
        
        # Calculate timing metrics (if available)
        timings = []
        for log in execution_logs:
            # Extract timing information if available
            if 'execution_time' in log:
                timings.append(log['execution_time'])
        
        metrics = {
            'total_actions': len(execution_logs),
            'success_rate': sum(1 for log in execution_logs if log.get('execution_success', False)) / len(execution_logs)
        }
        
        if timings:
            metrics.update({
                'average_execution_time': sum(timings) / len(timings),
                'min_execution_time': min(timings),
                'max_execution_time': max(timings)
            })
        
        return metrics
    
    def _combine_insights(self, visual_analysis: Dict[str, Any], execution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine insights from visual and execution analysis."""
        
        combined = {
            'overall_success_rate': 0,
            'key_insights': [],
            'potential_issues': [],
            'recommendations': []
        }
        
        # Calculate overall success rate
        if 'execution_analysis' in execution_analysis and execution_analysis.get('total_actions', 0) > 0:
            combined['overall_success_rate'] = execution_analysis['successful_actions'] / execution_analysis['total_actions']
        
        # Generate key insights
        if visual_analysis.get('total_frames', 0) > 0:
            combined['key_insights'].append(f"Episode captured {visual_analysis['total_frames']} visual frames")
        
        if execution_analysis.get('total_actions', 0) > 0:
            combined['key_insights'].append(f"Executed {execution_analysis['total_actions']} actions with {combined['overall_success_rate']:.1%} success rate")
        
        # Identify potential issues
        anomalies = visual_analysis.get('visual_anomalies', [])
        for anomaly in anomalies:
            if anomaly['severity'] in ['high', 'medium']:
                combined['potential_issues'].append(anomaly['description'])
        
        error_patterns = execution_analysis.get('error_patterns', [])
        for error in error_patterns[:3]:  # Top 3 errors
            combined['potential_issues'].append(f"Error in {error.get('subtask', 'unknown')}: {error.get('error', 'unknown error')}")
        
        # Generate recommendations
        if combined['overall_success_rate'] < 0.8:
            combined['recommendations'].append("Consider improving action success rates through better UI element targeting")
        
        if len(anomalies) > 3:
            combined['recommendations'].append("Address visual anomalies to improve test reliability")
        
        if len(error_patterns) > 2:
            combined['recommendations'].append("Implement better error handling and recovery mechanisms")
        
        return combined
    
    def _generate_overall_assessment(self, visual_analysis: Dict[str, Any], execution_analysis: Dict[str, Any], llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of the episode."""
        
        # Calculate assessment score
        score = 0.0
        factors = []
        
        # Success rate factor
        success_rate = execution_analysis.get('success_rate', 0) if 'success_rate' in execution_analysis else 0.5
        score += success_rate * 0.4
        factors.append(('success_rate', success_rate))
        
        # Visual quality factor
        anomalies = visual_analysis.get('visual_anomalies', [])
        visual_quality = max(0, 1 - len(anomalies) * 0.1)
        score += visual_quality * 0.3
        factors.append(('visual_quality', visual_quality))
        
        # LLM confidence factor
        llm_confidence = llm_analysis.get('confidence_score', 0.5)
        score += llm_confidence * 0.3
        factors.append(('llm_confidence', llm_confidence))
        
        # Determine grade
        if score >= 0.8:
            grade = 'A'
        elif score >= 0.6:
            grade = 'B'
        elif score >= 0.4:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'overall_score': score,
            'grade': grade,
            'assessment_factors': factors,
            'summary': f"Episode achieved {grade} grade with {score:.1%} overall performance score"
        }


class VisualTraceRecorder:
    """
    Records visual traces from environment rendering for analysis.
    Captures frame-by-frame UI images using env.render(mode="rgb_array").
    """
    
    def __init__(self, output_dir: str = "visual_traces"):
        self.output_dir = output_dir
        self.current_episode_dir = None
        self.frame_count = 0
        self.trace_metadata = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def start_episode(self, episode_id: str, goal: str) -> str:
        """
        Start recording a new episode.
        
        Args:
            episode_id: Unique identifier for the episode
            goal: The goal/task being performed
            
        Returns:
            Path to the episode directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}_{timestamp}")
        os.makedirs(episode_dir, exist_ok=True)
        
        self.current_episode_dir = episode_dir
        self.frame_count = 0
        
        # Store episode metadata
        self.trace_metadata = {
            'episode_id': episode_id,
            'goal': goal,
            'start_time': datetime.now().isoformat(),
            'frame_count': 0,
            'frames': []
        }
        
        return episode_dir
    
    def record_frame(self, 
                    env, 
                    step_info: Dict[str, Any] = None,
                    agent_action: str = None) -> Dict[str, Any]:
        """
        Record a single frame from the environment.
        
        Args:
            env: Environment object with get_state method (Android environment)
            step_info: Additional information about the current step
            agent_action: Action being performed by the agent
            
        Returns:
            Frame metadata including path and timestamp
        """
        if not self.current_episode_dir:
            raise ValueError("No active episode. Call start_episode() first.")
        
        try:
            # For Android environment, use get_state() instead of render()
            if hasattr(env, 'get_state'):
                # Android environment - get state with pixels
                state = env.get_state(wait_to_stabilize=True)
                frame = state.pixels  # RGB array from State object
            elif hasattr(env, 'render'):
                # Standard gym environment - use render method
                frame = env.render(mode="rgb_array")
            else:
                print("Warning: Environment has neither get_state() nor render() method")
                return None
            
            if frame is None:
                print("Warning: Could not capture frame")
                return None
            
            # Convert numpy array to PIL Image
            if isinstance(frame, np.ndarray):
                # Handle different color formats
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # RGB format
                    pil_image = Image.fromarray(frame)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    # RGBA format - convert to RGB
                    pil_image = Image.fromarray(frame[:, :, :3])
                else:
                    # Grayscale or other format
                    pil_image = Image.fromarray(frame)
            else:
                print(f"Warning: Unexpected frame format: {type(frame)}")
                return None
            
            # Generate frame filename
            frame_filename = f"frame_{self.frame_count:06d}.png"
            frame_path = os.path.join(self.current_episode_dir, frame_filename)
            
            # Save frame
            pil_image.save(frame_path, "PNG")
            
            # Create frame metadata
            frame_metadata = {
                'frame_number': self.frame_count,
                'timestamp': datetime.now().isoformat(),
                'filename': frame_filename,
                'path': frame_path,
                'agent_action': agent_action,
                'step_info': step_info or {},
                'image_size': pil_image.size,
                'image_mode': pil_image.mode
            }
            
            # Store frame metadata
            self.trace_metadata['frames'].append(frame_metadata)
            self.trace_metadata['frame_count'] = self.frame_count + 1
            
            self.frame_count += 1
            
            return frame_metadata
            
        except Exception as e:
            print(f"Error recording frame: {e}")
            return None
    
    def record_frame_with_ui_state(self, 
                                  env, 
                                  ui_state: Dict[str, Any],
                                  agent_action: str = None) -> Dict[str, Any]:
        """
        Record a frame along with UI state information.
        
        Args:
            env: Environment object with get_state method (Android environment)
            ui_state: Current UI state information
            agent_action: Action being performed by the agent
            
        Returns:
            Frame metadata with UI state
        """
        frame_metadata = self.record_frame(env, ui_state, agent_action)
        
        if frame_metadata:
            # Add UI state information
            frame_metadata['ui_state'] = {
                'ui_elements': ui_state.get('ui_elements', []),
                'ui_tree': ui_state.get('ui_tree', ''),
                'screen_title': ui_state.get('screen_title', ''),
                'current_activity': ui_state.get('current_activity', ''),
                'summary': ui_state.get('summary', ''),
                'subtask_info': ui_state.get('subtask_info', {}),
                'execution_result': ui_state.get('execution_result', {})
            }
        
        return frame_metadata
    
    def end_episode(self, final_outcome: str = "unknown") -> Dict[str, Any]:
        """
        End the current episode and save metadata.
        
        Args:
            final_outcome: Final outcome of the episode
            
        Returns:
            Episode summary
        """
        if not self.current_episode_dir:
            raise ValueError("No active episode to end.")
        
        # Update metadata
        self.trace_metadata['end_time'] = datetime.now().isoformat()
        self.trace_metadata['final_outcome'] = final_outcome
        self.trace_metadata['total_frames'] = self.frame_count
        
        # Save metadata to file
        metadata_path = os.path.join(self.current_episode_dir, "trace_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.trace_metadata, f, indent=2)
        
        # Create episode summary
        episode_summary = {
            'episode_id': self.trace_metadata['episode_id'],
            'goal': self.trace_metadata['goal'],
            'start_time': self.trace_metadata['start_time'],
            'end_time': self.trace_metadata['end_time'],
            'total_frames': self.frame_count,
            'final_outcome': final_outcome,
            'trace_directory': self.current_episode_dir,
            'metadata_file': metadata_path
        }
        
        # Reset for next episode
        self.current_episode_dir = None
        self.frame_count = 0
        self.trace_metadata = {}
        
        return episode_summary
    
    def get_frame_as_base64(self, frame_path: str) -> str:
        """
        Convert a frame image to base64 string for easy transmission.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(frame_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return ""
    
    def create_video_from_frames(self, episode_dir: str, output_filename: str = "episode_video.mp4") -> str:
        """
        Create a video from the recorded frames.
        
        Args:
            episode_dir: Directory containing the frames
            output_filename: Name of the output video file
            
        Returns:
            Path to the created video file
        """
        try:
            # Get all frame files
            frame_files = [f for f in os.listdir(episode_dir) if f.startswith('frame_') and f.endswith('.png')]
            frame_files.sort()
            
            if not frame_files:
                print("No frame files found")
                return None
            
            # Read first frame to get dimensions
            first_frame_path = os.path.join(episode_dir, frame_files[0])
            first_frame = cv2.imread(first_frame_path)
            height, width, layers = first_frame.shape
            
            # Create video writer
            output_path = os.path.join(episode_dir, output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
            
            # Add frames to video
            for frame_file in frame_files:
                frame_path = os.path.join(episode_dir, frame_file)
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
            
            video_writer.release()
            return output_path
            
        except Exception as e:
            print(f"Error creating video: {e}")
            return None


class EvaluationReportGenerator:
    """
    Generates comprehensive evaluation reports for agent performance analysis.
    Covers bug detection accuracy, agent recovery ability, and supervisor feedback effectiveness.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or MockLLMClient()
        self.evaluation_history = []
    
    def generate_evaluation_report(self, 
                                 episode_analysis: Dict[str, Any],
                                 execution_logs: List[Dict[str, Any]] = None,
                                 verifier_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            episode_analysis: Analysis results from EpisodeAnalyzer
            execution_logs: Raw execution logs from the episode
            verifier_logs: Verification logs from the verifier agent
            
        Returns:
            Comprehensive evaluation report
        """
        try:
            print("📊 Generating comprehensive evaluation report...")
            
            # Calculate evaluation metrics
            bug_detection_metrics = self._calculate_bug_detection_accuracy(episode_analysis, verifier_logs)
            recovery_metrics = self._calculate_agent_recovery_ability(episode_analysis, execution_logs)
            feedback_metrics = self._calculate_supervisor_feedback_effectiveness(episode_analysis)
            
            # Generate overall assessment
            overall_assessment = self._generate_overall_evaluation_assessment(
                bug_detection_metrics, recovery_metrics, feedback_metrics
            )
            
            # Create comprehensive report
            evaluation_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'episode_id': episode_analysis.get('episode_id'),
                    'evaluation_version': '1.0',
                    'report_type': 'comprehensive_evaluation'
                },
                'bug_detection_accuracy': bug_detection_metrics,
                'agent_recovery_ability': recovery_metrics,
                'supervisor_feedback_effectiveness': feedback_metrics,
                'overall_assessment': overall_assessment,
                'detailed_analysis': {
                    'visual_analysis_summary': episode_analysis.get('visual_analysis', {}),
                    'execution_analysis_summary': episode_analysis.get('execution_analysis', {}),
                    'llm_insights_summary': episode_analysis.get('llm_analysis', {})
                },
                'recommendations': self._generate_evaluation_recommendations(
                    bug_detection_metrics, recovery_metrics, feedback_metrics
                )
            }
            
            # Store in history
            self.evaluation_history.append(evaluation_report)
            
            print("✅ Evaluation report generated successfully")
            return evaluation_report
            
        except Exception as e:
            print(f"❌ Error generating evaluation report: {e}")
            return {'error': f'Evaluation report generation failed: {str(e)}'}
    
    def _calculate_bug_detection_accuracy(self, episode_analysis: Dict[str, Any], verifier_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate bug detection accuracy metrics."""
        
        metrics = {
            'overall_accuracy': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'detected_issues': [],
            'missed_issues': [],
            'false_alarms': []
        }
        
        try:
            # Analyze visual anomalies as potential bugs
            visual_analysis = episode_analysis.get('visual_analysis', {})
            anomalies = visual_analysis.get('visual_anomalies', [])
            
            # Analyze execution errors
            execution_analysis = episode_analysis.get('execution_analysis', {})
            error_patterns = execution_analysis.get('error_patterns', [])
            
            # Count detected issues
            detected_issues = []
            for anomaly in anomalies:
                if anomaly.get('severity') in ['high', 'medium']:
                    detected_issues.append({
                        'type': 'visual_anomaly',
                        'description': anomaly.get('description'),
                        'severity': anomaly.get('severity'),
                        'frame': anomaly.get('frame')
                    })
            
            for error in error_patterns:
                detected_issues.append({
                    'type': 'execution_error',
                    'description': error.get('error'),
                    'subtask': error.get('subtask'),
                    'action': error.get('action')
                })
            
            metrics['detected_issues'] = detected_issues
            metrics['true_positives'] = len(detected_issues)
            
            # Analyze verifier logs if available
            if verifier_logs:
                for log in verifier_logs:
                    if log.get('verdict') == 'FAILED':
                        metrics['true_positives'] += 1
                        detected_issues.append({
                            'type': 'verifier_failure',
                            'description': log.get('reason', 'Verification failed'),
                            'confidence': log.get('confidence', 0.0)
                        })
            
            # Calculate accuracy metrics
            total_issues = metrics['true_positives'] + metrics['false_negatives']
            if total_issues > 0:
                metrics['recall'] = metrics['true_positives'] / total_issues
            
            total_detections = metrics['true_positives'] + metrics['false_positives']
            if total_detections > 0:
                metrics['precision'] = metrics['true_positives'] / total_detections
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            
            metrics['overall_accuracy'] = metrics['f1_score']
            
        except Exception as e:
            print(f"Warning: Error calculating bug detection accuracy: {e}")
        
        return metrics
    
    def _calculate_agent_recovery_ability(self, episode_analysis: Dict[str, Any], execution_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate agent recovery ability metrics."""
        
        metrics = {
            'recovery_success_rate': 0.0,
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'recovery_patterns': [],
            'resilience_score': 0.0
        }
        
        try:
            # Analyze execution patterns for failures and recoveries
            execution_analysis = episode_analysis.get('execution_analysis', {})
            action_types = execution_analysis.get('action_types', {})
            
            total_actions = execution_analysis.get('total_actions', 0)
            failed_actions = execution_analysis.get('failed_actions', 0)
            
            metrics['total_failures'] = failed_actions
            
            # Analyze recovery patterns from execution logs
            if execution_logs:
                recovery_attempts = []
                for i, log in enumerate(execution_logs):
                    if not log.get('execution_success', True):
                        # Look for recovery attempts in subsequent logs
                        recovery_success = False
                        recovery_time = 0
                        
                        for j in range(i + 1, min(i + 5, len(execution_logs))):
                            next_log = execution_logs[j]
                            if next_log.get('execution_success', False):
                                recovery_success = True
                                recovery_time = j - i
                                break
                        
                        recovery_attempts.append({
                            'failure_index': i,
                            'recovery_success': recovery_success,
                            'recovery_time': recovery_time,
                            'failure_type': log.get('error_message', 'unknown')
                        })
                
                metrics['recovery_patterns'] = recovery_attempts
                metrics['successful_recoveries'] = sum(1 for r in recovery_attempts if r['recovery_success'])
                metrics['failed_recoveries'] = len(recovery_attempts) - metrics['successful_recoveries']
                
                if len(recovery_attempts) > 0:
                    metrics['recovery_success_rate'] = metrics['successful_recoveries'] / len(recovery_attempts)
                    
                    successful_times = [r['recovery_time'] for r in recovery_attempts if r['recovery_success']]
                    if successful_times:
                        metrics['average_recovery_time'] = sum(successful_times) / len(successful_times)
            
            # Calculate resilience score
            if total_actions > 0:
                success_rate = (total_actions - failed_actions) / total_actions
                recovery_bonus = metrics['recovery_success_rate'] * 0.3
                metrics['resilience_score'] = min(1.0, success_rate + recovery_bonus)
            
        except Exception as e:
            print(f"Warning: Error calculating agent recovery ability: {e}")
        
        return metrics
    
    def _calculate_supervisor_feedback_effectiveness(self, episode_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate supervisor feedback effectiveness metrics."""
        
        metrics = {
            'feedback_quality_score': 0.0,
            'insight_relevance': 0.0,
            'actionability_score': 0.0,
            'timeliness_score': 0.0,
            'improvement_suggestions_count': 0,
            'implemented_suggestions': 0,
            'feedback_categories': {
                'prompt_improvements': 0,
                'failure_patterns': 0,
                'coverage_recommendations': 0
            },
            'effectiveness_indicators': []
        }
        
        try:
            # Analyze LLM insights quality
            llm_analysis = episode_analysis.get('llm_analysis', {})
            
            # Count feedback categories
            prompt_improvements = llm_analysis.get('prompt_improvements', [])
            failure_patterns = llm_analysis.get('failure_patterns', [])
            coverage_recommendations = llm_analysis.get('coverage_recommendations', [])
            
            metrics['feedback_categories']['prompt_improvements'] = len(prompt_improvements)
            metrics['feedback_categories']['failure_patterns'] = len(failure_patterns)
            metrics['feedback_categories']['coverage_recommendations'] = len(coverage_recommendations)
            
            total_suggestions = len(prompt_improvements) + len(failure_patterns) + len(coverage_recommendations)
            metrics['improvement_suggestions_count'] = total_suggestions
            
            # Calculate insight relevance based on detected issues
            visual_analysis = episode_analysis.get('visual_analysis', {})
            execution_analysis = episode_analysis.get('execution_analysis', {})
            
            detected_issues = len(visual_analysis.get('visual_anomalies', [])) + len(execution_analysis.get('error_patterns', []))
            
            if detected_issues > 0 and total_suggestions > 0:
                # Higher relevance if suggestions match detected issues
                metrics['insight_relevance'] = min(1.0, total_suggestions / max(detected_issues, 1))
            else:
                metrics['insight_relevance'] = 0.5  # Neutral if no issues detected
            
            # Calculate actionability score
            actionable_suggestions = 0
            for suggestion in prompt_improvements + failure_patterns + coverage_recommendations:
                # Check if suggestion contains actionable keywords
                actionable_keywords = ['add', 'implement', 'improve', 'enhance', 'modify', 'update', 'change']
                if any(keyword in suggestion.lower() for keyword in actionable_keywords):
                    actionable_suggestions += 1
            
            if total_suggestions > 0:
                metrics['actionability_score'] = actionable_suggestions / total_suggestions
            
            # Calculate timeliness score (based on when feedback was provided)
            llm_confidence = llm_analysis.get('confidence_score', 0.5)
            metrics['timeliness_score'] = llm_confidence  # Higher confidence suggests better timing
            
            # Calculate overall feedback quality
            metrics['feedback_quality_score'] = (
                metrics['insight_relevance'] * 0.4 +
                metrics['actionability_score'] * 0.3 +
                metrics['timeliness_score'] * 0.3
            )
            
            # Generate effectiveness indicators
            if metrics['feedback_quality_score'] > 0.8:
                metrics['effectiveness_indicators'].append("High-quality feedback provided")
            elif metrics['feedback_quality_score'] > 0.6:
                metrics['effectiveness_indicators'].append("Moderate feedback quality")
            else:
                metrics['effectiveness_indicators'].append("Low feedback quality - needs improvement")
            
            if metrics['insight_relevance'] > 0.7:
                metrics['effectiveness_indicators'].append("Highly relevant insights")
            
            if metrics['actionability_score'] > 0.7:
                metrics['effectiveness_indicators'].append("Highly actionable suggestions")
            
        except Exception as e:
            print(f"Warning: Error calculating supervisor feedback effectiveness: {e}")
        
        return metrics
    
    def _generate_overall_evaluation_assessment(self, 
                                              bug_detection_metrics: Dict[str, Any],
                                              recovery_metrics: Dict[str, Any],
                                              feedback_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall evaluation assessment."""
        
        # Calculate composite scores
        bug_detection_score = bug_detection_metrics.get('overall_accuracy', 0.0)
        recovery_score = recovery_metrics.get('resilience_score', 0.0)
        feedback_score = feedback_metrics.get('feedback_quality_score', 0.0)
        
        # Weighted overall score
        overall_score = (
            bug_detection_score * 0.4 +
            recovery_score * 0.35 +
            feedback_score * 0.25
        )
        
        # Determine grade
        if overall_score >= 0.9:
            grade = 'A+'
            assessment = 'Excellent performance across all metrics'
        elif overall_score >= 0.8:
            grade = 'A'
            assessment = 'Very good performance with minor areas for improvement'
        elif overall_score >= 0.7:
            grade = 'B+'
            assessment = 'Good performance with some improvement opportunities'
        elif overall_score >= 0.6:
            grade = 'B'
            assessment = 'Satisfactory performance with notable improvement areas'
        elif overall_score >= 0.5:
            grade = 'C+'
            assessment = 'Below average performance requiring significant improvements'
        else:
            grade = 'C'
            assessment = 'Poor performance requiring major improvements'
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'assessment': assessment,
            'component_scores': {
                'bug_detection_accuracy': bug_detection_score,
                'agent_recovery_ability': recovery_score,
                'supervisor_feedback_effectiveness': feedback_score
            },
            'strengths': self._identify_strengths(bug_detection_metrics, recovery_metrics, feedback_metrics),
            'weaknesses': self._identify_weaknesses(bug_detection_metrics, recovery_metrics, feedback_metrics),
            'priority_improvements': self._identify_priority_improvements(
                bug_detection_metrics, recovery_metrics, feedback_metrics
            )
        }
    
    def _identify_strengths(self, bug_detection_metrics: Dict[str, Any], 
                           recovery_metrics: Dict[str, Any], 
                           feedback_metrics: Dict[str, Any]) -> List[str]:
        """Identify system strengths."""
        strengths = []
        
        if bug_detection_metrics.get('overall_accuracy', 0) > 0.8:
            strengths.append("High bug detection accuracy")
        
        if recovery_metrics.get('recovery_success_rate', 0) > 0.7:
            strengths.append("Strong agent recovery ability")
        
        if feedback_metrics.get('feedback_quality_score', 0) > 0.8:
            strengths.append("Effective supervisor feedback")
        
        if recovery_metrics.get('resilience_score', 0) > 0.8:
            strengths.append("High system resilience")
        
        if bug_detection_metrics.get('precision', 0) > 0.8:
            strengths.append("Low false positive rate in bug detection")
        
        return strengths
    
    def _identify_weaknesses(self, bug_detection_metrics: Dict[str, Any], 
                            recovery_metrics: Dict[str, Any], 
                            feedback_metrics: Dict[str, Any]) -> List[str]:
        """Identify system weaknesses."""
        weaknesses = []
        
        if bug_detection_metrics.get('overall_accuracy', 0) < 0.6:
            weaknesses.append("Low bug detection accuracy")
        
        if recovery_metrics.get('recovery_success_rate', 0) < 0.5:
            weaknesses.append("Poor agent recovery ability")
        
        if feedback_metrics.get('feedback_quality_score', 0) < 0.6:
            weaknesses.append("Ineffective supervisor feedback")
        
        if recovery_metrics.get('total_failures', 0) > 5:
            weaknesses.append("High failure rate")
        
        if feedback_metrics.get('actionability_score', 0) < 0.5:
            weaknesses.append("Low actionability of feedback suggestions")
        
        return weaknesses
    
    def _identify_priority_improvements(self, bug_detection_metrics: Dict[str, Any], 
                                      recovery_metrics: Dict[str, Any], 
                                      feedback_metrics: Dict[str, Any]) -> List[str]:
        """Identify priority improvements."""
        improvements = []
        
        # Prioritize based on lowest scores
        scores = [
            ('bug_detection', bug_detection_metrics.get('overall_accuracy', 0)),
            ('recovery', recovery_metrics.get('resilience_score', 0)),
            ('feedback', feedback_metrics.get('feedback_quality_score', 0))
        ]
        
        scores.sort(key=lambda x: x[1])  # Sort by score (lowest first)
        
        for area, score in scores[:2]:  # Top 2 priority areas
            if area == 'bug_detection' and score < 0.7:
                improvements.append("Improve bug detection algorithms and validation")
            elif area == 'recovery' and score < 0.7:
                improvements.append("Enhance agent recovery mechanisms and error handling")
            elif area == 'feedback' and score < 0.7:
                improvements.append("Strengthen supervisor feedback quality and relevance")
        
        return improvements
    
    def _generate_evaluation_recommendations(self, 
                                           bug_detection_metrics: Dict[str, Any],
                                           recovery_metrics: Dict[str, Any], 
                                           feedback_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific recommendations based on evaluation results."""
        
        recommendations = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_enhancements': []
        }
        
        # Immediate actions based on critical issues
        if bug_detection_metrics.get('overall_accuracy', 0) < 0.5:
            recommendations['immediate_actions'].append("Review and fix bug detection logic")
        
        if recovery_metrics.get('recovery_success_rate', 0) < 0.3:
            recommendations['immediate_actions'].append("Implement basic error recovery mechanisms")
        
        # Short-term improvements
        if bug_detection_metrics.get('precision', 0) < 0.7:
            recommendations['short_term_improvements'].append("Reduce false positive rate in bug detection")
        
        if recovery_metrics.get('average_recovery_time', 0) > 3:
            recommendations['short_term_improvements'].append("Optimize recovery time for faster error resolution")
        
        if feedback_metrics.get('actionability_score', 0) < 0.6:
            recommendations['short_term_improvements'].append("Improve actionability of supervisor feedback")
        
        # Long-term enhancements
        if bug_detection_metrics.get('overall_accuracy', 0) < 0.8:
            recommendations['long_term_enhancements'].append("Implement advanced bug detection algorithms")
        
        if recovery_metrics.get('resilience_score', 0) < 0.8:
            recommendations['long_term_enhancements'].append("Develop comprehensive resilience framework")
        
        if feedback_metrics.get('feedback_quality_score', 0) < 0.8:
            recommendations['long_term_enhancements'].append("Enhance supervisor feedback system with ML capabilities")
        
        return recommendations


class SupervisorAgent:
    """
    Supervisor Agent with visual trace recording and intelligent analysis capabilities.
    Records frame-by-frame UI images and analyzes test episodes using LLM-powered insights.
    """
    
    def __init__(self, llm_client=None, visual_trace_dir: str = "visual_traces"):
        self.llm_client = llm_client
        self.visual_recorder = VisualTraceRecorder(visual_trace_dir)
        self.episode_analyzer = EpisodeAnalyzer(llm_client)
        self.evaluation_generator = EvaluationReportGenerator(llm_client)
        self.episode_history = []
        self.improvement_suggestions = []
        self.current_episode_id = None
        self.current_episode_dir = None
    
    def start_episode_recording(self, episode_id: str, goal: str) -> str:
        """
        Start recording visual traces for a new episode.
        
        Args:
            episode_id: Unique identifier for the episode
            goal: The goal/task being performed
            
        Returns:
            Path to the episode directory
        """
        self.current_episode_id = episode_id
        self.current_episode_dir = self.visual_recorder.start_episode(episode_id, goal)
        return self.current_episode_dir
    
    def record_step(self, 
                   env, 
                   ui_state: Dict[str, Any] = None,
                   agent_action: str = None) -> Dict[str, Any]:
        """
        Record a step in the current episode with visual trace.
        
        Args:
            env: Environment object with render method
            ui_state: Current UI state information
            agent_action: Action being performed by the agent
            
        Returns:
            Frame metadata
        """
        if not self.current_episode_id:
            raise ValueError("No active episode. Call start_episode_recording() first.")
        
        return self.visual_recorder.record_frame_with_ui_state(env, ui_state, agent_action)
    
    def end_episode_recording(self, final_outcome: str = "unknown") -> Dict[str, Any]:
        """
        End the current episode recording and save all data.
        
        Args:
            final_outcome: Final outcome of the episode
            
        Returns:
            Episode summary
        """
        if not self.current_episode_id:
            raise ValueError("No active episode to end.")
        
        episode_summary = self.visual_recorder.end_episode(final_outcome)
        self.current_episode_id = None
        
        return episode_summary
    
    def get_episode_frames(self, episode_dir: str) -> List[Dict[str, Any]]:
        """
        Get all frames for a specific episode.
        
        Args:
            episode_dir: Directory containing the episode frames
            
        Returns:
            List of frame metadata
        """
        metadata_path = os.path.join(episode_dir, "trace_metadata.json")
        
        if not os.path.exists(metadata_path):
            return []
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('frames', [])
        except Exception as e:
            print(f"Error reading episode metadata: {e}")
            return []
    
    def create_episode_video(self, episode_dir: str) -> str:
        """
        Create a video from the episode frames.
        
        Args:
            episode_dir: Directory containing the episode frames
            
        Returns:
            Path to the created video file
        """
        return self.visual_recorder.create_video_from_frames(episode_dir)
    
    def get_visual_trace_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all recorded visual traces.
        
        Returns:
            Summary of visual trace recordings
        """
        if not os.path.exists(self.visual_recorder.output_dir):
            return {'message': 'No visual traces recorded yet'}
        
        episodes = []
        total_frames = 0
        
        for item in os.listdir(self.visual_recorder.output_dir):
            item_path = os.path.join(self.visual_recorder.output_dir, item)
            if os.path.isdir(item_path) and item.startswith('episode_'):
                metadata_path = os.path.join(item_path, "trace_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            episodes.append({
                                'episode_id': metadata.get('episode_id'),
                                'goal': metadata.get('goal'),
                                'total_frames': metadata.get('total_frames', 0),
                                'start_time': metadata.get('start_time'),
                                'final_outcome': metadata.get('final_outcome')
                            })
                            total_frames += metadata.get('total_frames', 0)
                    except Exception as e:
                        print(f"Error reading metadata for {item}: {e}")
        
        return {
            'total_episodes': len(episodes),
            'total_frames': total_frames,
            'episodes': episodes
        }
    
    def analyze_episode(self, execution_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the current episode using visual traces and execution logs.
        
        Args:
            execution_logs: List of execution logs from the episode
            
        Returns:
            Comprehensive episode analysis with LLM insights
        """
        if not self.current_episode_dir:
            raise ValueError("No active episode to analyze. Call start_episode_recording() first.")
        
        print("🔍 Starting comprehensive episode analysis...")
        
        # Analyze the episode
        analysis_result = self.episode_analyzer.analyze_episode_trace(
            self.current_episode_dir, 
            execution_logs
        )
        
        if 'error' in analysis_result:
            print(f"❌ Analysis failed: {analysis_result['error']}")
            return analysis_result
        
        # Store analysis in episode history
        self.episode_history.append(analysis_result)
        
        # Extract improvement suggestions
        llm_analysis = analysis_result.get('llm_analysis', {})
        self.improvement_suggestions.extend(llm_analysis.get('prompt_improvements', []))
        
        print("✅ Episode analysis completed")
        return analysis_result
    
    def get_prompt_improvements(self) -> List[str]:
        """
        Get prompt improvement suggestions from LLM analysis.
        
        Returns:
            List of prompt improvement suggestions
        """
        return self.improvement_suggestions
    
    def get_failure_patterns(self) -> List[str]:
        """
        Get identified failure patterns from LLM analysis.
        
        Returns:
            List of failure patterns
        """
        failure_patterns = []
        for episode in self.episode_history:
            llm_analysis = episode.get('llm_analysis', {})
            failure_patterns.extend(llm_analysis.get('failure_patterns', []))
        return failure_patterns
    
    def get_coverage_recommendations(self) -> List[str]:
        """
        Get test coverage expansion recommendations from LLM analysis.
        
        Returns:
            List of coverage recommendations
        """
        coverage_recommendations = []
        for episode in self.episode_history:
            llm_analysis = episode.get('llm_analysis', {})
            coverage_recommendations.extend(llm_analysis.get('coverage_recommendations', []))
        return coverage_recommendations
    
    def generate_analysis_report(self, episode_dir: str = None, execution_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report for an episode.
        
        Args:
            episode_dir: Directory containing episode data (uses current if None)
            execution_logs: Execution logs for the episode
            
        Returns:
            Comprehensive analysis report
        """
        if episode_dir is None:
            episode_dir = self.current_episode_dir
        
        if not episode_dir:
            raise ValueError("No episode directory specified")
        
        print("📊 Generating comprehensive analysis report...")
        
        # Analyze the episode
        analysis = self.episode_analyzer.analyze_episode_trace(episode_dir, execution_logs)
        
        if 'error' in analysis:
            return analysis
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'episode_dir': episode_dir,
                'analysis_version': '1.0'
            },
            'episode_summary': {
                'episode_id': analysis.get('episode_id'),
                'goal': analysis.get('goal'),
                'overall_assessment': analysis.get('overall_assessment', {})
            },
            'visual_analysis': analysis.get('visual_analysis', {}),
            'execution_analysis': analysis.get('execution_analysis', {}),
            'llm_insights': {
                'prompt_improvements': analysis.get('llm_analysis', {}).get('prompt_improvements', []),
                'failure_patterns': analysis.get('llm_analysis', {}).get('failure_patterns', []),
                'coverage_recommendations': analysis.get('llm_analysis', {}).get('coverage_recommendations', [])
            },
            'recommendations': {
                'immediate_actions': self._generate_immediate_actions(analysis),
                'long_term_improvements': self._generate_long_term_improvements(analysis)
            }
        }
        
        # Save report to file
        report_filename = f"supervisor_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(episode_dir, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Analysis report saved to: {report_path}")
        
        return report
    
    def _generate_immediate_actions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate immediate action recommendations based on analysis."""
        actions = []
        
        # Check for high-severity issues
        visual_analysis = analysis.get('visual_analysis', {})
        anomalies = visual_analysis.get('visual_anomalies', [])
        
        high_severity_anomalies = [a for a in anomalies if a.get('severity') == 'high']
        if high_severity_anomalies:
            actions.append(f"Address {len(high_severity_anomalies)} high-severity visual anomalies")
        
        # Check for execution failures
        execution_analysis = analysis.get('execution_analysis', {})
        if execution_analysis.get('failed_actions', 0) > 0:
            actions.append(f"Investigate {execution_analysis['failed_actions']} failed actions")
        
        # Check for low success rates
        overall_assessment = analysis.get('overall_assessment', {})
        if overall_assessment.get('overall_score', 1.0) < 0.6:
            actions.append("Review and improve action success rates")
        
        return actions
    
    def _generate_long_term_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate long-term improvement recommendations based on analysis."""
        improvements = []
        
        # Get LLM recommendations
        llm_analysis = analysis.get('llm_analysis', {})
        improvements.extend(llm_analysis.get('prompt_improvements', []))
        improvements.extend(llm_analysis.get('coverage_recommendations', []))
        
        # Add performance improvements
        execution_analysis = analysis.get('execution_analysis', {})
        if execution_analysis.get('total_actions', 0) > 10:
            improvements.append("Consider optimizing action sequences for efficiency")
        
        # Add reliability improvements
        visual_analysis = analysis.get('visual_analysis', {})
        if len(visual_analysis.get('visual_anomalies', [])) > 5:
            improvements.append("Implement more robust error handling and recovery mechanisms")
        
        return improvements
    
    def get_supervisor_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all supervisor activities.
        
        Returns:
            Summary of supervisor activities and insights
        """
        total_episodes = len(self.episode_history)
        
        if total_episodes == 0:
            return {'message': 'No episodes analyzed yet'}
        
        # Calculate overall statistics
        total_frames = sum(ep.get('visual_analysis', {}).get('total_frames', 0) for ep in self.episode_history)
        avg_score = sum(ep.get('overall_assessment', {}).get('overall_score', 0) for ep in self.episode_history) / total_episodes
        
        # Collect all insights
        all_prompt_improvements = self.get_prompt_improvements()
        all_failure_patterns = self.get_failure_patterns()
        all_coverage_recommendations = self.get_coverage_recommendations()
        
        return {
            'total_episodes_analyzed': total_episodes,
            'total_frames_captured': total_frames,
            'average_performance_score': avg_score,
            'total_insights_generated': {
                'prompt_improvements': len(all_prompt_improvements),
                'failure_patterns': len(all_failure_patterns),
                'coverage_recommendations': len(all_coverage_recommendations)
            },
            'recent_insights': {
                'prompt_improvements': all_prompt_improvements[-5:] if all_prompt_improvements else [],
                'failure_patterns': all_failure_patterns[-5:] if all_failure_patterns else [],
                'coverage_recommendations': all_coverage_recommendations[-5:] if all_coverage_recommendations else []
            },
            'episode_history': self.episode_history
        }
    
    def create_evaluation_report(self, 
                               execution_logs: List[Dict[str, Any]] = None,
                               verifier_logs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report for the current episode.
        
        Args:
            execution_logs: Raw execution logs from the episode
            verifier_logs: Verification logs from the verifier agent
            
        Returns:
            Comprehensive evaluation report
        """
        if not self.current_episode_dir:
            raise ValueError("No active episode to evaluate. Call start_episode_recording() first.")
        
        print("📊 Creating comprehensive evaluation report...")
        
        # First analyze the episode if not already done
        episode_analysis = self.analyze_episode(execution_logs)
        
        if 'error' in episode_analysis:
            return episode_analysis
        
        # Generate evaluation report
        evaluation_report = self.evaluation_generator.generate_evaluation_report(
            episode_analysis, execution_logs, verifier_logs
        )
        
        if 'error' in evaluation_report:
            return evaluation_report
        
        # Save evaluation report to file
        report_filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(self.current_episode_dir, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        print(f"📄 Evaluation report saved to: {report_path}")
        
        return evaluation_report
    
    def get_bug_detection_accuracy(self) -> Dict[str, Any]:
        """
        Get bug detection accuracy metrics from the latest evaluation.
        
        Returns:
            Bug detection accuracy metrics
        """
        if not self.evaluation_generator.evaluation_history:
            return {'message': 'No evaluation reports available'}
        
        latest_evaluation = self.evaluation_generator.evaluation_history[-1]
        return latest_evaluation.get('bug_detection_accuracy', {})
    
    def get_agent_recovery_ability(self) -> Dict[str, Any]:
        """
        Get agent recovery ability metrics from the latest evaluation.
        
        Returns:
            Agent recovery ability metrics
        """
        if not self.evaluation_generator.evaluation_history:
            return {'message': 'No evaluation reports available'}
        
        latest_evaluation = self.evaluation_generator.evaluation_history[-1]
        return latest_evaluation.get('agent_recovery_ability', {})
    
    def get_supervisor_feedback_effectiveness(self) -> Dict[str, Any]:
        """
        Get supervisor feedback effectiveness metrics from the latest evaluation.
        
        Returns:
            Supervisor feedback effectiveness metrics
        """
        if not self.evaluation_generator.evaluation_history:
            return {'message': 'No evaluation reports available'}
        
        latest_evaluation = self.evaluation_generator.evaluation_history[-1]
        return latest_evaluation.get('supervisor_feedback_effectiveness', {})
    
    def get_overall_evaluation_grade(self) -> Dict[str, Any]:
        """
        Get overall evaluation grade and assessment.
        
        Returns:
            Overall evaluation assessment
        """
        if not self.evaluation_generator.evaluation_history:
            return {'message': 'No evaluation reports available'}
        
        latest_evaluation = self.evaluation_generator.evaluation_history[-1]
        return latest_evaluation.get('overall_assessment', {})
    
    def generate_comprehensive_evaluation_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all evaluation reports.
        
        Returns:
            Comprehensive evaluation summary
        """
        if not self.evaluation_generator.evaluation_history:
            return {'message': 'No evaluation reports available'}
        
        evaluations = self.evaluation_generator.evaluation_history
        
        # Calculate aggregate metrics
        total_evaluations = len(evaluations)
        avg_overall_score = sum(e.get('overall_assessment', {}).get('overall_score', 0) for e in evaluations) / total_evaluations
        
        # Aggregate component scores
        avg_bug_detection = sum(e.get('bug_detection_accuracy', {}).get('overall_accuracy', 0) for e in evaluations) / total_evaluations
        avg_recovery = sum(e.get('agent_recovery_ability', {}).get('resilience_score', 0) for e in evaluations) / total_evaluations
        avg_feedback = sum(e.get('supervisor_feedback_effectiveness', {}).get('feedback_quality_score', 0) for e in evaluations) / total_evaluations
        
        # Collect all recommendations
        all_recommendations = []
        for evaluation in evaluations:
            recommendations = evaluation.get('recommendations', {})
            all_recommendations.extend(recommendations.get('immediate_actions', []))
            all_recommendations.extend(recommendations.get('short_term_improvements', []))
            all_recommendations.extend(recommendations.get('long_term_enhancements', []))
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return {
            'total_evaluations': total_evaluations,
            'average_scores': {
                'overall_score': avg_overall_score,
                'bug_detection_accuracy': avg_bug_detection,
                'agent_recovery_ability': avg_recovery,
                'supervisor_feedback_effectiveness': avg_feedback
            },
            'performance_trend': self._analyze_performance_trend(evaluations),
            'top_recommendations': unique_recommendations[:10],  # Top 10 recommendations
            'evaluation_history': [
                {
                    'episode_id': e.get('report_metadata', {}).get('episode_id'),
                    'overall_score': e.get('overall_assessment', {}).get('overall_score'),
                    'grade': e.get('overall_assessment', {}).get('grade'),
                    'generated_at': e.get('report_metadata', {}).get('generated_at')
                }
                for e in evaluations
            ]
        }
    
    def _analyze_performance_trend(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends across evaluations."""
        
        if len(evaluations) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Extract scores over time
        scores = []
        for evaluation in evaluations:
            overall_score = evaluation.get('overall_assessment', {}).get('overall_score', 0)
            generated_at = evaluation.get('report_metadata', {}).get('generated_at')
            scores.append({
                'score': overall_score,
                'timestamp': generated_at
            })
        
        # Calculate trend
        if len(scores) >= 2:
            first_score = scores[0]['score']
            last_score = scores[-1]['score']
            score_change = last_score - first_score
            
            if score_change > 0.1:
                trend = 'improving'
            elif score_change < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend_direction': trend,
            'score_change': score_change if len(scores) >= 2 else 0,
            'consistency': self._calculate_consistency(scores),
            'best_performance': max(s['score'] for s in scores),
            'worst_performance': min(s['score'] for s in scores)
        }
    
    def _calculate_consistency(self, scores: List[Dict[str, Any]]) -> str:
        """Calculate performance consistency."""
        
        if len(scores) < 2:
            return 'insufficient_data'
        
        score_values = [s['score'] for s in scores]
        variance = sum((score - sum(score_values) / len(score_values)) ** 2 for score in score_values) / len(score_values)
        std_dev = variance ** 0.5
        
        if std_dev < 0.1:
            return 'high'
        elif std_dev < 0.2:
            return 'medium'
        else:
            return 'low' 