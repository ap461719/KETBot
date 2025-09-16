import os
import logging
import uuid
import importlib
from typing import Optional

from arklex.env.tools.tools import Tool
from arklex.env.planner.function_calling import FunctionCallingPlanner
from arklex.utils.graph_state import StatusEnum
from arklex.orchestrator.NLU.nlu import SlotFilling


logger = logging.getLogger(__name__)

class BaseResourceInitializer:
    @staticmethod
    def init_tools(tools):
        raise NotImplementedError

    @staticmethod
    def init_workers(workers):
        raise NotImplementedError
    
class DefaulResourceInitializer(BaseResourceInitializer):
    @staticmethod
    def init_tools(tools):
        # return dict of valid tools with name and description
        tool_registry = {}
        for tool in tools:
            tool_id = tool["id"]
            name = tool["name"]
            path = tool["path"]
            try: # try to import the tool to check its existance
                filepath = os.path.join("arklex.env.tools", path)
                module_name = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func = getattr(module, name)
            except Exception as e:
                logger.error(f"Tool {name} is not registered, error: {e}")
                continue
            tool_registry[tool_id] = {
                "name": func().name,
                "description": func().description,
                "execute": func,
                "fixed_args": tool.get("fixed_args", {}),
            }
        return tool_registry
    
    @staticmethod
    def init_workers(workers):
        worker_registry = {}
        for worker in workers:
            worker_id = worker["id"]
            name = worker["name"]
            path = worker["path"]
            #print("Path in ENV.PY:")
            #print(path)
            try: # try to import the worker to check its existance
                #filepath = os.path.join("arklex.env.workers", path)
                #filepath = f"arklex.env.{path.replace('/', '.')}".rstrip(".py")
                #print("I'M IMPORTING THE WORKERS")
                filepath = path 
                module_name = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func = getattr(module, name)
            except Exception as e:
                logger.error(f"Worker {name} is not registered, error: {e}")
                continue
            worker_registry[worker_id] = {
                "name": name,
                "description": func().description,
                "execute": func
            }
        #print(f"\n[DEBUG] 🛠️ Registered Workers: {worker_registry.keys()}")
        return worker_registry

class Env():
    def __init__(self, tools, workers, slotsfillapi, resource_inizializer: Optional[BaseResourceInitializer] = None):
        if resource_inizializer is None:
            resource_inizializer = DefaulResourceInitializer()
        self.tools = resource_inizializer.init_tools(tools)
        self.workers = resource_inizializer.init_workers(workers)
        self.name2id = {resource["name"]: id for id, resource in {**self.tools, **self.workers}.items()}
        self.id2name = {id: resource["name"] for id, resource in {**self.tools, **self.workers}.items()}
        self.slotfillapi = self.initialize_slotfillapi(slotsfillapi)
        self.planner = FunctionCallingPlanner(
            tools_map=self.tools,
            name2id=self.name2id
        )

    def initialize_slotfillapi(self, slotsfillapi):
        return SlotFilling(slotsfillapi)

    def step(self, id, message_state, params):
        #print(f"[DEBUG] 🛠️ Type of message_state INSIDE step(): {type(message_state)}")
        #print("EXAMINER TURN INSIDE STEP()")
        #print(message_state.get("examiner_turn", "MISSING"))

        #if id is None or id not in self.id2name:
            #print("IN ENV.PY: this is the examiner_turn status when id is None:")
            #print(message_state.get("examiner_turn", False))
            #print(f"[ERROR] ❌ Expected Worker Not Assigned! Using Default Worker.")
            #id = "ExaminerWorker" if message_state.get("examiner_turn", False) else "StudentWorker"
            

        #print(f"\n[DEBUG] 🚀 step() called with ID: {id} | Expected Worker: {self.id2name.get(id, 'Unknown')}")
        #print(f"[DEBUG] 🔍 Message State: {message_state}")
        #print(f"[DEBUG] 🔄 Examiner Turn: {message_state.get('examiner_turn')}")
        #print(f"[DEBUG] 🔄 Available Workers: {list(self.workers.keys())}")
        #print(f"[DEBUG] 🔄 Action Passed to step(): {self.id2name.get(id)}")

        #print(f"[DEBUG] 🛠️ Type of message_state INSIDE step(): {type(message_state)}")
        #print("EXAMINER TURN INSIDE STEP()")
        #print(message_state.get("examiner_turn", "MISSING"))
        #print(f"[DEBUG] 🛠️ Inside step(), Message State Received: {message_state}")
        #print(f"[DEBUG] 🔄 Examiner Turn inside step(): {message_state.get('examiner_turn', 'MISSING')}")
        #print(f"[DEBUG] 🔄 Stage inside step(): {message_state.get('stage', 'MISSING')}")
        #print(f"[DEBUG] 🔄 Current Topic inside step(): {message_state.get('current_topic', 'MISSING')}")
        #print(f"[DEBUG] 🔄 Addressed Student inside step(): {message_state.get('addressed_student', 'MISSING')}")
        #print(f"[DEBUG] 🔄 Previous Student Response inside step(): {message_state.get('previous_student_response', 'MISSING')}")
        #print(f"[DEBUG] 🔄 Difficulty inside step(): {message_state.get('difficulty', 'MISSING')}")


        # Extract examiner_turn value
        examiner_turn = message_state.get("examiner_turn", False)

        # Dynamically get the correct worker IDs
        examiner_worker_id = next(
            (worker_id for worker_id, worker in self.workers.items() if worker["name"] == "ExaminerWorker"), 
            None
        )
        student_worker_id = next(
            (worker_id for worker_id, worker in self.workers.items() if worker["name"] == "StudentWorker"), 
            None
        )

        # 🔹 Ensure correct worker is used even if `id` is already set
        if examiner_turn and id != examiner_worker_id:
            #print("[DEBUG] 🔄 Overriding Worker to ExaminerWorker because examiner_turn=True")
            id = examiner_worker_id
        elif not examiner_turn and id != student_worker_id:
            #print("[DEBUG] 🔄 Overriding Worker to StudentWorker because examiner_turn=False")
            id = student_worker_id
  

        if id in self.tools:
            logger.info(f"{self.tools[id]['name']} tool selected")
            tool: Tool = self.tools[id]["execute"]()
            tool.init_slotfilling(self.slotfillapi)
            response_state = tool.execute(message_state, **self.tools[id]["fixed_args"])
            params["history"] = response_state.get("trajectory", [])
            params["dialog_states"] = response_state.get("slots", [])
            current_node = params.get("curr_node")
            params["node_status"][current_node] = response_state.get("status", StatusEnum.COMPLETE.value)
                
        elif id in self.workers:
            message_state["metadata"]["worker"] = self.workers
            logger.info(f"{self.workers[id]['name']} worker selected")
            #print(f"\n[DEBUG] 📌 Worker Selected: {self.workers[id]['name']} | ID: {id}")
            worker = self.workers[id]["execute"]()
            response_state = worker.execute(message_state)
            # ✅ If the examiner just spoke, save the question for the student
            if examiner_turn and response_state.get("response"):
                message_state["examiner_question"] = response_state["response"]
                #print(f"[DEBUG] ✅ Saved examiner_question for next turn: {message_state['examiner_question']}")
            call_id = str(uuid.uuid4())
            params["history"].append({'content': None, 'role': 'assistant', 'tool_calls': [{'function': {'arguments': "", 'name': self.id2name[id]}, 'id': call_id, 'type': 'function'}], 'function_call': None})
            params["history"].append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": self.id2name[id],
                        "content": response_state["response"]
            })
        else:
            logger.info("planner selected")
            action, response_state, msg_history = self.planner.execute(message_state, params["history"])
        
        logger.info(f"Response state from {id}: {response_state}")
        return response_state, params
