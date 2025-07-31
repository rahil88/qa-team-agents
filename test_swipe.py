#!/usr/bin/env python3
"""
Simple test script to debug swipe directions
"""

import time
from android_world.env import env_launcher
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent

# Set up environment
env = env_launcher.load_and_setup_env(
    console_port=5554,
    emulator_setup=False,
    freeze_datetime=True,
    adb_path="/Users/craddy-san/Library/Android/sdk/platform-tools/adb",
    grpc_port=8554,
)

grounding_agent = AndroidEnvGroundingAgent(env)

def test_swipe_direction(direction_name, direction):
    """Test a specific swipe direction"""
    print(f"\n🧪 TESTING: {direction_name}")
    print(f"📱 Action: swipe with direction='{direction}'")
    
    # Get initial state
    initial_state = env.get_state()
    initial_elements = len(initial_state.ui_elements)
    print(f"📊 Initial UI elements: {initial_elements}")
    
    # Perform swipe
    action = {"action_type": "swipe", "direction": direction}
    obs, reward, done, info = grounding_agent.step(action)
    
    # Wait and check result
    time.sleep(2)
    final_state = env.get_state()
    final_elements = len(final_state.ui_elements)
    print(f"📊 Final UI elements: {final_elements}")
    
    # Check for notification panel indicators
    notification_indicators = 0
    for element in final_state.ui_elements:
        text = getattr(element, 'text', None) or ''
        desc = getattr(element, 'content_description', None) or ''
        
        if any(term in text.lower() or term in desc.lower() for term in 
               ['notification', 'quick settings', 'clear all', 'brightness', 'wifi', 'bluetooth']):
            notification_indicators += 1
    
    print(f"🔍 Notification indicators found: {notification_indicators}")
    
    if notification_indicators > 0:
        print(f"✅ SUCCESS: {direction_name} opened notification panel!")
        return True
    else:
        print(f"❌ FAILED: {direction_name} did not open notification panel")
        return False

def main():
    print("🚀 SWIPE DIRECTION TEST")
    print("=" * 50)
    
    # Test different swipe directions
    results = {}
    
    # Test 1: swipe direction="down"
    results['down'] = test_swipe_direction("Swipe Direction DOWN", "down")
    time.sleep(3)  # Wait between tests
    
    # Test 2: swipe direction="up" 
    results['up'] = test_swipe_direction("Swipe Direction UP", "up")
    time.sleep(3)
    
    # Test 3: Try the opposite if both failed
    if not results['down'] and not results['up']:
        print("\n🔄 Both standard directions failed, trying alternatives...")
        
        # Try specific coordinates-based approach
        print(f"\n🧪 TESTING: Coordinate-based swipe")
        print(f"📱 Action: touch with drag from top to bottom")
        
        # Touch and drag approach
        touch_action = {
            "action_type": "touch",
            "touch_position": [0.5, 0.1]  # Start near top
        }
        grounding_agent.step(touch_action)
        time.sleep(0.5)
        
        drag_action = {
            "action_type": "touch", 
            "touch_position": [0.5, 0.5]  # Drag to middle
        }
        grounding_agent.step(drag_action)
        
        time.sleep(2)
        coord_state = env.get_state()
        coord_indicators = sum(1 for element in coord_state.ui_elements
                             if any(term in (getattr(element, 'text', '') or '').lower() or 
                                   term in (getattr(element, 'content_description', '') or '').lower()
                                   for term in ['notification', 'quick settings', 'clear all']))
        
        print(f"🔍 Coordinate approach - notification indicators: {coord_indicators}")
        results['coordinates'] = coord_indicators > 0
    
    # Summary
    print(f"\n📊 FINAL RESULTS:")
    print("=" * 30)
    for method, success in results.items():
        status = "✅ WORKS" if success else "❌ FAILS"
        print(f"{method:15} : {status}")
    
    # Recommendation
    working_methods = [method for method, success in results.items() if success]
    if working_methods:
        print(f"\n🎯 RECOMMENDATION: Use {working_methods[0]} for notification panel")
    else:
        print(f"\n⚠️ WARNING: No working swipe method found!")

if __name__ == "__main__":
    main() 