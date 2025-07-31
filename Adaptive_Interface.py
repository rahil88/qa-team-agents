#!/usr/bin/env python3
"""
Adaptive Interface for mid-execution replanning
Provides a clean interface for Planner to get adaptive recommendations from Verifier
"""

from typing import Dict, Any, List, Optional
from Verifier_Agent import VerifierAgent

class AdaptiveInterface:
    """
    Interface for Planner to interact with Verifier for mid-execution adaptation
    """
    
    def __init__(self, verifier: VerifierAgent):
        self.verifier = verifier
        
    def check_for_problems(self) -> Dict[str, Any]:
        """
        Check current UI state for problems requiring adaptation
        Returns dict with problems and recommendations
        """
        return self.verifier.get_adaptive_recommendations()
    
    def should_adapt(self, priority_threshold: str = "medium") -> bool:
        """
        Determine if adaptation is needed based on problem severity
        """
        recommendations = self.check_for_problems()
        
        if not recommendations['has_problems']:
            return False
            
        # Check if any problems meet the priority threshold
        priority_levels = {"low": 1, "medium": 2, "high": 3, "immediate": 4}
        threshold_level = priority_levels.get(priority_threshold, 2)
        
        for problem in recommendations['problems']:
            problem_level = priority_levels.get(problem['severity'], 1)
            if problem_level >= threshold_level:
                return True
                
        return False
    
    def get_immediate_actions(self) -> List[Dict[str, Any]]:
        """
        Get immediate actions that should be executed before continuing
        """
        recommendations = self.check_for_problems()
        immediate_actions = []
        
        for rec in recommendations.get('recommendations', []):
            if rec['priority'] == 'immediate':
                immediate_actions.append(rec['specific_action'])
                
        return immediate_actions
    
    def get_alternative_strategies(self) -> List[Dict[str, Any]]:
        """
        Get alternative strategies when current approach fails
        """
        recommendations = self.check_for_problems()
        alternatives = []
        
        for rec in recommendations.get('recommendations', []):
            if rec['action_type'] in ['alternative_approach', 'retry_navigation']:
                alternatives.append({
                    'strategy': rec['action_type'],
                    'description': rec['description'],
                    'actions': rec.get('alternative_actions', [rec['specific_action']]),
                    'follow_up_steps': rec.get('follow_up_steps', [])
                })
                
        return alternatives
    
    def get_problem_summary(self) -> Dict[str, Any]:
        """
        Get a summary of detected problems for logging/debugging
        """
        recommendations = self.check_for_problems()
        
        problem_counts = {'high': 0, 'medium': 0, 'low': 0}
        problem_types = []
        
        for problem in recommendations.get('problems', []):
            severity = problem['severity']
            if severity in problem_counts:
                problem_counts[severity] += 1
            problem_types.append(problem['type'])
            
        return {
            'total_problems': len(recommendations.get('problems', [])),
            'severity_breakdown': problem_counts,
            'problem_types': problem_types,
            'requires_immediate_action': recommendations.get('priority_level') == 'immediate',
            'has_alternatives': len(self.get_alternative_strategies()) > 0
        }

# Example usage patterns for Planner integration:
"""
Usage Examples:

# 1. Basic problem checking
adaptive_interface = AdaptiveInterface(verifier)
if adaptive_interface.should_adapt("high"):
    immediate_actions = adaptive_interface.get_immediate_actions()
    for action in immediate_actions:
        executor.execute_action(action)

# 2. Alternative strategy when stuck
if current_approach_failed:
    alternatives = adaptive_interface.get_alternative_strategies()
    if alternatives:
        new_strategy = alternatives[0]
        planner.update_strategy(new_strategy)

# 3. Problem monitoring during execution
problem_summary = adaptive_interface.get_problem_summary()
if problem_summary['requires_immediate_action']:
    logger.warning(f"Immediate action required: {problem_summary}")
""" 