from gui_agents.s2.agents.manager import Manager
from android_world.agents.agent_s_android import AndroidEnvGroundingAgent
from android_world.env import env_launcher
import os
from gui_agents.utils import download_kb_data
import json

# Only download knowledge base if it doesn't exist to preserve our enhancements
kb_dir = "/Users/craddy-san/Desktop/untitled folder/kb_s2"
if not os.path.exists(kb_dir) or not os.path.exists(os.path.join(kb_dir, "android")):
    print("📥 Downloading initial knowledge base...")
    download_kb_data(
        version="s2",                # or "s1" for Agent S1
        release_tag="v0.2.2",        # or the latest release tag
        download_dir=kb_dir,
        platform="darwin"            # or "linux", "windows", "android"
    )
else:
    print("✅ Using existing enhanced knowledge base...")


# 1. Engine params
engine_params = {
    "engine_type": "openai",
    "openai_api_key": os.environ.get("OPENAI_API_KEY"),
    "model": "gpt-3.5-turbo"
}

# 2. Android environment and grounding agent
env = env_launcher.load_and_setup_env(
    console_port=5554,
    emulator_setup=False,
    freeze_datetime=True,
    adb_path="/Users/craddy-san/Library/Android/sdk/platform-tools/adb",
    grpc_port=8554,
)
controller = env.controller
grounding_agent = AndroidEnvGroundingAgent(controller)

# 3. Knowledge base path
local_kb_path = "/Users/craddy-san/Desktop/untitled folder/kb_s2"

# 4. Embedding engine - temporarily disable for testing
# from gui_agents.s2.core.engine import OpenAIEmbeddingEngine
# embedding_engine = OpenAIEmbeddingEngine(engine_params)
embedding_engine = None  # Disable embeddings for testing core improvements

# Create a Planner class that can be instantiated
class Planner:
    def __init__(self):
        # 1. Engine params
        self.engine_params = {
            "engine_type": "openai",
            "openai_api_key": os.environ.get("OPENAI_API_KEY"),
            "model": "gpt-3.5-turbo"
        }
        
        # 2. Android environment and grounding agent
        self.env = env_launcher.load_and_setup_env(
            console_port=5554,
            emulator_setup=False,
            freeze_datetime=True,
            adb_path="/Users/craddy-san/Library/Android/sdk/platform-tools/adb",
            grpc_port=8554,
        )
        self.controller = self.env.controller
        self.grounding_agent = AndroidEnvGroundingAgent(self.controller)
        
        # 3. Knowledge base path
        self.local_kb_path = "/Users/craddy-san/Desktop/untitled folder/kb_s2"
        
        # 4. Embedding engine - temporarily disable for testing
        self.embedding_engine = None  # Disable embeddings for testing core improvements
        
        # 5. Instantiate the planner
        self.planner = Manager(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path=self.local_kb_path,
            embedding_engine=self.embedding_engine,
            platform="android" 
        )
    
    def get_action_queue(self, instruction, observation=None):
        """Get action queue for a given instruction."""
        if observation is None:
            timestep = self.controller.reset()
            observation = self.grounding_agent._convert_timestep_to_obs(timestep)
        
        planner_info, subtasks = self.planner.get_action_queue(
            instruction=instruction,
            observation=observation,
            failed_subtask=None,
            completed_subtasks_list=[],
            remaining_subtasks_list=[]
        )
        
        return planner_info, subtasks
    
    def generate_plan(self, goal: str, current_state=None) -> list:
        """
        Generate a plan for the given goal.
        
        Args:
            goal: The goal to achieve
            current_state: Current state information (optional)
            
        Returns:
            List of subtasks as dictionaries
        """
        try:
            # Get action queue using the existing method
            planner_info, subtasks = self.get_action_queue(goal)
            
            # Convert Node objects to dictionaries for the orchestrator
            plan = []
            for node in subtasks:
                subtask_dict = {
                    "name": node.name,
                    "info": node.info,
                    "type": "action"
                }
                plan.append(subtask_dict)
            
            return plan
            
        except Exception as e:
            print(f"❌ Error generating plan: {e}")
            return []

# Create a global instance for backward compatibility
planner = Planner()

if __name__ == "__main__":
    # Test the planner
    instruction = "Turn on Wi-Fi from Settings"
    planner_info, subtasks = planner.get_action_queue(instruction)
    
    print("Planner Info:", planner_info)
    print("Subtasks:", subtasks)
    
    # Convert Node objects to dicts
    serializable_subtasks = [
        {"name": node.name, "info": node.info}  # add more fields if your Node has them
        for node in subtasks
    ]
    
    with open("subtasks.json", "w") as f:
        json.dump(serializable_subtasks, f)


