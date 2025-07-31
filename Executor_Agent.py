import os
import json
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
from android_world.env import env_launcher
import time

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

class SmartExecutor:
    """
    Smart executor that leverages AndroidEnvGroundingAgent's UI interaction capabilities
    """
    def __init__(self, grounding_agent):
        self.agent = grounding_agent
        
    def execute_subtasks(self, subtasks):
        """Execute subtasks using the grounding agent's built-in capabilities"""
        for subtask in subtasks:
            print(f"\n🎯 Executing subtask: {subtask['name']}")
            self._execute_single_subtask(subtask)
            
    def _execute_single_subtask(self, subtask):
        """Execute a single subtask - one simple action based on subtask name"""
        subtask_name = subtask["name"].lower()
        
        print(f"📱 Processing: {subtask_name}")
        
        # Convert the subtask name directly to a single action
        action = self._convert_subtask_name_to_action(subtask_name)
        
        if action:
            print(f"🔄 Action: {action}")
            # Use grounding agent's step method - it handles everything!
            obs, reward, done, info = self.agent.step(action)
            
            # Improved wait timing based on action type
            if "wait" in subtask_name:
                time.sleep(3)  # Wait longer for explicit wait commands
            elif "swipe" in subtask_name:
                time.sleep(1)  # Short wait for swipe actions to complete
            elif "open" in subtask_name and "app" in subtask_name:
                time.sleep(6)  # Longer wait for app launches (Settings takes time)
                # Force UI state refresh for app launches
                print("🔄 Forcing UI state refresh after app launch...")
                for i in range(2):
                    time.sleep(1)
                    state = env.get_state(wait_to_stabilize=True)
                    print(f"   UI refresh {i+1}/2 completed")
            else:
                time.sleep(2)  # Normal wait for other actions
            
            # Verify the action succeeded by checking current screen
            self._verify_action_success(subtask_name)
            
        else:
            print("⚠️ Could not convert subtask to action")
            
    def _verify_action_success(self, subtask_name):
        """Verify that the action actually succeeded"""
        # Force UI refresh by waiting for UI to stabilize
        time.sleep(1.0)  # Allow UI animation to complete
        
        # Try multiple state refreshes to ensure we get the latest UI
        ui_elements = []
        for refresh_attempt in range(3):
            print(f"🔄 UI state refresh attempt {refresh_attempt + 1}/3...")
            # Get state with wait_to_stabilize=True to ensure fresh UI
            state = env.get_state(wait_to_stabilize=True)
            ui_elements = getattr(state, 'ui_elements', [])
            
            # Convert ui_elements to dict format if needed
            if ui_elements and not isinstance(ui_elements[0], dict):
                ui_elements = self._convert_ui_elements_to_dict(ui_elements)
            
            # Check if we have any meaningful UI elements
            if ui_elements and len(ui_elements) > 0:
                # Look for any text content to see if UI has changed
                text_elements = [elem.get('text', '') for elem in ui_elements if elem.get('text')]
                print(f"   Found {len(text_elements)} text elements: {text_elements[:3]}...")
                
                # If we're looking for Settings and see Settings-related text, we might be good
                if "open settings app" in subtask_name:
                    settings_related = any('settings' in text.lower() for text in text_elements)
                    if settings_related:
                        print("   ✅ Found Settings-related text elements")
                        break
                    elif refresh_attempt < 2:
                        print("   ⏳ Waiting for Settings UI to appear...")
                        time.sleep(2)
                        continue
            else:
                print("   ⚠️ No UI elements found")
                if refresh_attempt < 2:
                    time.sleep(2)
                    continue
        
        print(f"🔍 Verifying success for: {subtask_name}")
        print(f"📊 Found {len(ui_elements)} UI elements in verification")
        
        # Debug: Print a few key elements to see what we have
        print("🔍 Key UI elements after action:")
        for i, element in enumerate(ui_elements[:5]):  # Show first 5 elements
            text = element.get('text') or ''
            desc = element.get('content_description') or ''
            print(f"  [{i}] text='{text}' desc='{desc}'")
        
        # Check if we're in the expected screen state
        if "open settings app" in subtask_name:
            # After opening Settings app, we should see Settings menu items
            settings_indicators = ["network & internet", "connected devices", "apps", "notifications", "battery", "display", "sound", "storage"]
            found = any(self._find_element_by_text(ui_elements, indicator) 
                       for indicator in settings_indicators)
            if not found:
                print("⚠️ Settings app not detected, trying to tap Settings again...")
                # Retry tapping settings
                action = self._create_tap_action("settings")
                if action:
                    self.agent.step(action)
                    time.sleep(6)  # Increased wait time for Settings to load
                    # Force multiple UI refreshes
                    for attempt in range(3):
                        print(f"🔄 UI refresh attempt {attempt + 1}/3...")
                        state = env.get_state(wait_to_stabilize=True)
                        ui_elements = getattr(state, 'ui_elements', [])
                        if ui_elements and not isinstance(ui_elements[0], dict):
                            ui_elements = self._convert_ui_elements_to_dict(ui_elements)
                        
                        # Check if we now have Settings elements
                        found = any(self._find_element_by_text(ui_elements, indicator) 
                                   for indicator in settings_indicators)
                        if found:
                            print(f"✅ Settings app opened successfully on attempt {attempt + 1}!")
                            break
                        else:
                            print(f"❌ Still on home screen after attempt {attempt + 1}")
                            if attempt < 2:  # Don't sleep on last attempt
                                time.sleep(2)
                    else:
                        print("❌ Settings app still not detected after all retry attempts")
                        # Try a different approach - tap Settings icon again with different coordinates
                        print("🔄 Trying alternative Settings tap...")
                        # Try tapping the Settings icon with a slight offset
                        action = {"action_type": "touch", "touch_position": [0.16, 0.68]}  # Slightly different coordinates
                        self.agent.step(action)
                        time.sleep(4)
                        state = env.get_state(wait_to_stabilize=True)
                        ui_elements = getattr(state, 'ui_elements', [])
                        if ui_elements and not isinstance(ui_elements[0], dict):
                            ui_elements = self._convert_ui_elements_to_dict(ui_elements)
                        found = any(self._find_element_by_text(ui_elements, indicator) 
                                   for indicator in settings_indicators)
                        if found:
                            print("✅ Settings app opened with alternative tap!")
                        else:
                            print("❌ All Settings app attempts failed")
                    
        elif "tap settings" in subtask_name:
            # After tapping settings, we should see settings menu items
            settings_indicators = ["network & internet", "connected devices", "apps", "notifications", "battery"]
            found = any(self._find_element_by_text(ui_elements, indicator) 
                       for indicator in settings_indicators)
            if not found:
                print("⚠️ Settings screen not detected, trying to tap Settings again...")
                # Retry tapping settings
                action = self._create_tap_action("settings")
                if action:
                    self.agent.step(action)
                    time.sleep(4)  # Increased wait time
                    # Get fresh state after retry
                    state = env.get_state(wait_to_stabilize=True)
                    ui_elements = getattr(state, 'ui_elements', [])
                    if ui_elements and not isinstance(ui_elements[0], dict):
                        ui_elements = self._convert_ui_elements_to_dict(ui_elements)
                    
        elif "network" in subtask_name and "internet" in subtask_name:
            # Should see network options
            network_indicators = ["internet","", "airplane mode", "hotspot", "vpn", "wi-fi", "mobile network"]
            found = any(self._find_element_by_text(ui_elements, indicator) 
                       for indicator in network_indicators)
            if not found:
                print("⚠️ Network & Internet screen not detected")
                print("🔄 Attempting to navigate to Network & Internet...")
                # Try tapping the fallback position or look for alternative text
                action = self._create_flexible_tap_action(["network & internet", "network", "connections"])
                if action:
                    self.agent.step(action)
                    time.sleep(3)
                
        elif "internet" in subtask_name:
            # Should see internet/wifi options
            internet_indicators = ["wi-fi", "wifi", "mobile network", "data usage"]
            found = any(self._find_element_by_text(ui_elements, indicator) 
                       for indicator in internet_indicators)
            if not found:
                print("⚠️ Internet screen not detected")
                
    def _parse_steps_from_info(self, info):
        """Parse step-by-step instructions from subtask info"""
        if not info:
            return []
            
        # Split by numbered steps (1. 2. 3. etc.)
        lines = info.split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering (1. 2. etc.) and clean up
                step = line.split('.', 1)[-1].strip()
                if step:
                    steps.append(step)
                    
        return steps
    
    def _convert_subtask_name_to_action(self, subtask_name):
        """Convert Pixel 6 subtask name directly to action"""
        
        # Settings app actions
        if "open settings app" in subtask_name or "tap settings" in subtask_name:
            return self._create_tap_action("settings")
            
        elif "tap network internet" in subtask_name:
            # Try multiple possible texts for network settings
            return self._create_flexible_tap_action(["network & internet", "network", "connections", "connectivity"])
            
        elif "tap internet" in subtask_name:
            # Try multiple possible texts for internet settings  
            return self._create_flexible_tap_action(["internet", "mobile network", "data", "network"])
            
        elif "tap wi-fi toggle" in subtask_name or "tap wifi toggle" in subtask_name:
            # On Pixel 6, Wi-Fi is accessed through "Internet" in Network & Internet
            return self._create_tap_action("internet")
            
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
        
    def _create_wifi_tile_action(self):
        """Create action specifically for Wi-Fi tile in quick settings"""
        state = env.get_state(wait_to_stabilize=True)
        ui_elements = getattr(state, 'ui_elements', [])
        
        # Convert ui_elements to dict format if needed
        if ui_elements and not isinstance(ui_elements[0], dict):
            ui_elements = self._convert_ui_elements_to_dict(ui_elements)
        
        print("🔍 Looking for Wi-Fi tile in quick settings...")
        print(f"📊 Total UI elements found: {len(ui_elements)}")
        
        # DEBUG: Print all UI elements
        print("🔎 ALL UI ELEMENTS:")
        for i, element in enumerate(ui_elements):
            text = element.get('text') or ''
            desc = element.get('content_description') or ''
            clickable = element.get('is_clickable', False)
            bounds = element.get('bounds', [])
            print(f"  [{i:2}] text='{text}' desc='{desc}' clickable={clickable} bounds={bounds}")
        
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
                    bounds = element.get('bounds')
                    if bounds:
                        x = (bounds[0] + bounds[2]) / 2
                        y = (bounds[1] + bounds[3]) / 2
                        
                        # Get screen size
                        try:
                            screen_width, screen_height = env.logical_screen_size
                        except:
                            screen_width, screen_height = 1080, 2400
                        
                        norm_x = x / screen_width
                        norm_y = y / screen_height
                        
                        print(f"🎯 Wi-Fi tile coordinates: ({int(x)}, {int(y)}) -> ({norm_x:.3f}, {norm_y:.3f})")
                        return {
                            "action_type": "touch",
                            "touch_position": [norm_x, norm_y]
                        }
        
        # If no exact match, look for elements containing network-related text
        print("🔍 No exact match found, trying partial matches...")
        for element in ui_elements:
            text = (element.get('text') or '').lower()
            desc = (element.get('content_description') or '').lower()
            element_text = text + ' ' + desc
            
            # Look for any network/internet indicators
            if any(pattern.lower() in element_text for pattern in ['internet', 'network', 'wifi', 'wireless', 'mobile']):
                if element.get('is_clickable', False):
                    print(f"✅ Found Wi-Fi tile partial match: '{element.get('text', '')}' / '{element.get('content_description', '')}'")
                    bounds = element.get('bounds')
                    if bounds:
                        x = (bounds[0] + bounds[2]) / 2
                        y = (bounds[1] + bounds[3]) / 2
                        
                        try:
                            screen_width, screen_height = env.logical_screen_size
                        except:
                            screen_width, screen_height = 1080, 2400
                        
                        norm_x = x / screen_width
                        norm_y = y / screen_height
                        
                        print(f"🎯 Wi-Fi tile coordinates: ({int(x)}, {int(y)}) -> ({norm_x:.3f}, {norm_y:.3f})")
                        return {
                            "action_type": "touch",
                            "touch_position": [norm_x, norm_y]
                        }
        
        print("❌ Could not find Wi-Fi/Internet tile in quick settings")
        print("🔍 Trying any clickable elements as last resort...")
        
        # Last resort: find any clickable elements
        clickable_elements = [el for el in ui_elements if el.get('is_clickable', False)]
        print(f"📱 Found {len(clickable_elements)} clickable elements")
        for i, element in enumerate(clickable_elements):
            text = element.get('text') or ''
            desc = element.get('content_description') or ''
            print(f"  Clickable[{i}]: '{text}' / '{desc}'")
        
        # Return a wait action instead of falling back to random elements
        return {"action_type": "wait"}
    
    def _convert_step_to_action(self, step):
        """Convert a step description to an action the grounding agent can execute"""
        step_lower = step.lower()
        
        # Map common step patterns to actions
        if "tap" in step_lower and "settings" in step_lower:
            return self._create_tap_action("settings")
            
        elif "tap" in step_lower and ("network" in step_lower or "internet" in step_lower):
            return self._create_tap_action("network")
            
        elif "tap" in step_lower and ("wi-fi" in step_lower or "wifi" in step_lower):
            return self._create_tap_action("wi-fi")
            
        elif "toggle" in step_lower and ("wi-fi" in step_lower or "wifi" in step_lower):
            return self._create_tap_action("wi-fi")
            
        elif "swipe down" in step_lower:
            return {"action_type": "swipe", "direction": "down"}
            
        elif "swipe up" in step_lower:
            return {"action_type": "swipe", "direction": "up"}
            
        elif "open" in step_lower:
            # Extract app name
            app_name = self._extract_app_name(step)
            if app_name:
                return self._create_tap_action(app_name)
                
        elif "type" in step_lower or "enter" in step_lower:
            # Extract text to type
            text = self._extract_text_to_type(step)
            if text:
                return {"action_type": "type", "text": text}
                
        elif "wait" in step_lower or "pause" in step_lower:
            return {"action_type": "wait"}
            
        # If we can't parse it, try to find any clickable text mentioned
        return self._create_smart_tap_action(step)
        
    def _create_flexible_tap_action(self, target_texts):
        """Create a tap action by trying multiple possible text matches"""
        for target_text in target_texts:
            action = self._create_tap_action(target_text)
            # If we found a valid target (not the fallback), use it
            if action and action.get("touch_position") != [0.5, 0.5]:
                return action
        
        # If none found, return fallback
        print(f"⚠️ None of the target texts found: {target_texts}")
        return {"action_type": "touch", "touch_position": [0.5, 0.5]}
        
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
        
        if target and target.get('bounds'):
            # Calculate center coordinates
            bounds = target['bounds']
            x = (bounds[0] + bounds[2]) / 2
            y = (bounds[1] + bounds[3]) / 2
            
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
        
        # Fallback: try to click by approximate location
        return {"action_type": "touch", "touch_position": [0.5, 0.5]}
        
    def _create_smart_tap_action(self, step):
        """Smart action creation that tries to find any mentioned UI element"""
        # Extract potential UI element names from the step
        keywords = ["settings", "wi-fi", "wifi", "network", "internet", "toggle", "switch", "button"]
        
        for keyword in keywords:
            if keyword in step.lower():
                return self._create_tap_action(keyword)
                
        # Default fallback
        return {"action_type": "wait"}
        
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
                    # Handle None values properly
                    text = element.get('text') or ''
                    desc = element.get('content_description') or ''
                    element_text = (text + ' ' + desc).lower()
                    if keyword in element_text and element.get('is_clickable', False):
                        print(f"✅ Found keyword match '{keyword}': '{element.get('text', '')}'")
                        return element
        
        print(f"❌ No match found for '{target_text}'")
        return None
        
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
        
    def _extract_app_name(self, step):
        """Extract app name from step like 'open Settings app'"""
        step_lower = step.lower()
        if "settings" in step_lower:
            return "settings"
        elif "chrome" in step_lower:
            return "chrome"
        elif "camera" in step_lower:
            return "camera"
        # Add more apps as needed
        return None
        
    def _extract_text_to_type(self, step):
        """Extract text to type from step description"""
        # Simple extraction - you can make this more sophisticated
        if "type" in step.lower():
            # Look for quoted text
            import re
            quoted = re.findall(r'"([^"]*)"', step)
            if quoted:
                return quoted[0]
        return ""

# Main execution function that replaces the old execute_subtasks
def execute_subtasks(subtasks):
    """Main function that uses the smart executor"""
    executor = SmartExecutor(grounding_agent)
    executor.execute_subtasks(subtasks)

# Example usage
if __name__ == "__main__":
    with open("subtasks.json", "r") as f:
        my_subtasks = json.load(f)
    execute_subtasks(my_subtasks)
