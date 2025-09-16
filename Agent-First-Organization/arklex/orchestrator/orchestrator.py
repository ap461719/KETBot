import json
import time
from typing import Any, Dict
import logging
import uuid
import os
from typing import List, Dict, Any, Tuple
import ast
import copy
from arklex.env.env import Env
import janus
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
import langsmith as ls
from openai import OpenAI
from litellm import completion

from arklex.orchestrator.task_graph import TaskGraph
from arklex.env.tools.utils import ToolGenerator
from arklex.orchestrator.NLU.nlu import SlotFilling
from arklex.orchestrator.prompts import RESPOND_ACTION_NAME, RESPOND_ACTION_FIELD_NAME, REACT_INSTRUCTION
from arklex.types import EventType, StreamType
from arklex.utils.graph_state import ConvoMessage, OrchestratorMessage, MessageState, StatusEnum, BotConfig
from arklex.utils.utils import init_logger, format_chat_history
from arklex.orchestrator.NLU.nlu import NLU
from arklex.utils.trace import TraceRunName
from arklex.utils.model_config import MODEL


load_dotenv()
logger = logging.getLogger(__name__)


class AgentOrg:
    def __init__(self, config, env: Env, **kwargs):
        if isinstance(config, dict):
            self.product_kwargs = config
        else:
            self.product_kwargs = json.load(open(config))
        self.user_prefix = "user"
        self.worker_prefix = "assistant"
        self.environment_prefix = "tool"
        self.__eos_token = "\n"
        self.task_graph = TaskGraph("taskgraph", self.product_kwargs)
        self.env = env

    def generate_next_step(
        self, messages: List[Dict[str, Any]], message_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str, float]:
        res = completion(
                messages=messages,
                model=MODEL["model_type_or_path"],
                custom_llm_provider="openai",
                temperature=0.0
            )
        message = res.choices[0].message
        action_str = message.content.split("Action:")[-1].strip()
        #print(f"[DEBUG] 📥 Raw Action String: {action_str}")  # ADD THIS
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            print(f"[ERROR] ❌ Failed to Parse Action: {action_str}")  # ADD THIS
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        #print(f"[DEBUG] ✅ Parsed Action: {action_parsed}")  # ADD THIS
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = action_parsed["name"]
        if action is None:
            #print(f"[ERROR] ❌ `generate_next_step()` returned None! Assigning Default Worker.")
            action = "ExaminerWorker" if message_state.get("examiner_turn", False) else "StudentWorker"
            #print(f"[INFO] 🔄 Assigned Default Action: {action}")

        return message.model_dump(), action, res._hidden_params["response_cost"]


    def get_response(self, inputs: dict, stream_type: StreamType = None, message_queue: janus.SyncQueue = None) -> Dict[str, Any]:
        text = inputs["text"]
        try:
            message_state = json.loads(inputs["text"])  # Convert back to dictionary
            #print(f"[DEBUG] ✅ Deserialized message_state: {message_state}")  # Print after conversion
            #print(f"[DEBUG] 🔎 examiner_turn in get_response(): {message_state.get('examiner_turn', 'MISSING')}")
        except json.JSONDecodeError:
            print("[ERROR] ❌ Failed to deserialize message_state from JSON!")
        chat_history = inputs["chat_history"]
        params = inputs["parameters"]
        params["timing"] = {}
        chat_history_copy = copy.deepcopy(chat_history)
        chat_history_copy.append({"role": self.user_prefix, "content": text})
        chat_history_str = format_chat_history(chat_history_copy)
        params["dialog_states"] = params.get("dialog_states", {})
        metadata = params.get("metadata", {})
        metadata["chat_id"] = metadata.get("chat_id", str(uuid.uuid4()))
        metadata["turn_id"] = metadata.get("turn_id", 0) + 1
        metadata["tool_response"] = {}
        params["metadata"] = metadata
        params["history"] = params.get("history", [])
        if not params["history"]:
            params["history"] = copy.deepcopy(chat_history_copy)
        else:
            params["history"].append(chat_history_copy[-2])
            params["history"].append(chat_history_copy[-1])

        ##### Model safety checking
        # check the response, decide whether to give template response or not
        client = OpenAI()
        text = inputs["text"]
        moderation_response = client.moderations.create(input=text).model_dump()
        is_flagged = moderation_response["results"][0]["flagged"]
        if is_flagged:
            return_response = {
                "answer": self.product_kwargs["safety_response"],
                "parameters": params,
                "has_follow_up": True
            }
            return return_response

        ##### TaskGraph Chain
        taskgraph_inputs = {
            "text": text,
            "chat_history_str": chat_history_str,
            "parameters": params  ## TODO: different params for different components
        }
        dt = time.time()
        taskgraph_chain = RunnableLambda(self.task_graph.get_node) | RunnableLambda(self.task_graph.postprocess_node)
        node_info, params = taskgraph_chain.invoke(taskgraph_inputs)
        params["timing"]["taskgraph"] = time.time() - dt
        logger.info("=============node_info=============")
        logger.info(f"The first node info is : {node_info}") # {'name': 'MessageWorker', 'attribute': {'value': 'If you are interested, you can book a calendly meeting https://shorturl.at/crFLP with us. Or, you can tell me your phone number, email address, and name; our expert will reach out to you soon.', 'direct': False, 'slots': {"<name>": {<attributes>}}}}
        node_status = params.get("node_status", {})
        params["node_status"] = node_status

        with ls.trace(name=TraceRunName.TaskGraph, inputs={"taskgraph_inputs": taskgraph_inputs}) as rt:
            rt.end(
                outputs={
                    "metadata": params.get("metadata"),
                    "timing": params.get("timing", {}),
                    "curr_node": {
                        "id": params.get("curr_node"),
                        "name": node_info.get("name"),
                        "attribute": node_info.get("attribute")
                    },
                    "curr_global_intent": params.get("curr_pred_intent"),
                    "dialog_states": params.get("dialog_states"),
                    "node_status": params.get("node_status")}, 
                metadata={"chat_id": metadata.get("chat_id"), "turn_id": metadata.get("turn_id")}
            )

        # Direct response
        node_attribute = node_info["attribute"]
        if node_attribute["value"].strip():
            if node_attribute.get("direct_response"):                    
                return_response = {
                    "answer": node_attribute["value"],
                    "parameters": params
                }
                if node_attribute["type"] == "multiple-choice":
                    return_response["choice_list"] = node_attribute["choice_list"]
                return return_response

        # Tool/Worker
        user_message = ConvoMessage(history=chat_history_str, message=text)
        orchestrator_message = OrchestratorMessage(message=node_info["attribute"]["value"], attribute=node_info["attribute"])
        sys_instruct = "You are a " + self.product_kwargs["role"] + ". " + self.product_kwargs["user_objective"] + self.product_kwargs["builder_objective"] + self.product_kwargs["intro"] + self.product_kwargs.get("opt_instruct", "")
        logger.info("=============sys_instruct=============")
        logger.info(sys_instruct)
        bot_config = BotConfig(
            bot_id=self.product_kwargs.get("bot_id", "default"),
            version=self.product_kwargs.get("version", "default"),
            language=self.product_kwargs.get("language", "EN"),
            bot_type=self.product_kwargs.get("bot_type", "presalebot"),
            available_workers=self.product_kwargs.get("workers", [])
        )
        message_state = MessageState(
            sys_instruct=sys_instruct, 
            bot_config=bot_config,
            user_message=user_message, 
            orchestrator_message=orchestrator_message, 
            trajectory=params["history"], 
            message_flow=params.get("worker_response", {}).get("message_flow", ""), 
            slots=params.get("dialog_states"),
            metadata=params.get("metadata"),
            is_stream=True if stream_type is not None else False,
            message_queue=message_queue
        )
        #print(f"[DEBUG] 🛠️ Type of message_state BEFORE step(): {type(message_state)}")
        #print(f"[DEBUG] 📢 Examiner Turn status BEFORE step():")
        #print(message_state.get("examiner_turn", "MISSING"))
        #print("[DEBUG] Full message_state BEFORE step():", message_state)

        if "user_message" in message_state and isinstance(message_state["user_message"], ConvoMessage):
            try:
                user_message_data = json.loads(message_state["user_message"].message)  # Convert JSON string to dict
                message_state["examiner_turn"] = user_message_data.get("examiner_turn", False)  # Set examiner_turn
                message_state["stage"] = user_message_data.get("stage", "stage_1")
                message_state["current_topic"] = user_message_data.get("current_topic", "unknown")
                message_state["addressed_student"] = user_message_data.get("addressed_student", "unknown")
                message_state["previous_student_response"] = user_message_data.get("previous_student_response", "")
                message_state["difficulty"] = user_message_data.get("difficulty", "Easy")
                message_state["examiner_question"] = user_message_data.get("examiner_question", "")
                message_state["stage_2_conversation"] = user_message_data.get("stage_2_conversation", "")
                message_state["is_first_speaker"] = user_message_data.get("is_first_speaker", "")
                #message_state["subtopic_phase"] = user_message_data.get("subtopic_phase", 0)
                message_state["discussion_topic"] = user_message_data.get("discussion_topic", "")

                #print("This is the discussion topic in orchestrator.py")
                #print(message_state["discussion_topic"])

                #print("[DEBUG] ✅ Extracted Values BEFORE step():", message_state)
            except json.JSONDecodeError:
                print("[ERROR] Failed to parse user_message.message JSON.")

        
        #print(f"[DEBUG] 📢 Examiner Turn status BEFORE step() AFTER PARSING:")
        #print(message_state.get("examiner_turn", "MISSING"))

        #print(f"[DEBUG] 📜 Message State Before Step: {json.dumps(message_state, indent=2)}")

        response_state, params = self.env.step(node_info["id"], message_state, params)
        if message_state.get("examiner_turn", False) and response_state.get("response"):
            #print(f"[🛑 EARLY RETURN] Examiner Question = {response_state['response']}")
            return {
                "answer": response_state["response"],
                "tool_response": response_state,
                "parameters": params
            }

        
        logger.info(f"{response_state=}")

        tool_response = params.get("metadata", {}).get("tool_response", {})
        params["metadata"]["tool_response"] = {}

        with ls.trace(name=TraceRunName.ExecutionResult, inputs={"message_state": message_state}) as rt:
            rt.end(
                outputs={"metadata": params.get("metadata"), **response_state}, 
                metadata={"chat_id": metadata.get("chat_id"), "turn_id": metadata.get("turn_id")}
            )

        # ReAct framework to decide whether return to user or continue
        FINISH = False
        while not FINISH:
            # if the last response is from the assistant with content(which means not from tool or worker but from function calling response), 
            # then directly return the response otherwise it will continue to the next node but treat the previous response has been return to user.
            if response_state.get("trajectory", []) \
                and response_state["trajectory"][-1]["role"] == "assistant" \
                and response_state["trajectory"][-1]["content"]: 
                response_state["response"] = response_state["trajectory"][-1]["content"]
                break
            
            # If the current node is not complete, then no need to continue to the next node
            node_status = params.get("node_status", {})
            curr_node = params.get("curr_node", None)
            status = node_status.get(curr_node, StatusEnum.COMPLETE.value)
            if status == StatusEnum.INCOMPLETE.value:
                break

            node_info, params = taskgraph_chain.invoke(taskgraph_inputs)
            logger.info("=============node_info=============")
            logger.info(f"The while node info is : {node_info}")
            if node_info["id"] not in self.env.workers and node_info["id"] not in self.env.tools:
                #examiner_turn = message_state.get("examiner_turn", False)
                #print("ENTERED THE IF STATEMENT OIN ORCHESTRATOR.PY")
                #print(node_info["id"])
                message_state = MessageState(
                    sys_instruct=sys_instruct, 
                    user_message=user_message, 
                    orchestrator_message=orchestrator_message, 
                    trajectory=params["history"], 
                    message_flow=params.get("worker_response", {}).get("message_flow", ""), 
                    slots=params.get("dialog_states"),
                    metadata=params.get("metadata"),
                    is_stream=True if stream_type is not None else False,
                    message_queue=message_queue
                )
                
                action, response_state, msg_history = self.env.planner.execute(message_state, params["history"])
                params["history"] = msg_history
                if action == RESPOND_ACTION_NAME:
                    FINISH = True
                else:
                    tool_response = {}
            else:
                if node_info["id"] in self.env.tools:
                    node_actions = [{"name": self.env.id2name[node_info["id"]], "arguments": self.env.tools[node_info["id"]]["execute"]().info}]
                elif node_info["id"] in self.env.workers:
                    node_actions = [{"name": self.env.id2name[node_info["id"]], "description": self.env.workers[node_info["id"]]["execute"]().description}]
                action_spaces = node_actions
                action_spaces.append({"name": RESPOND_ACTION_NAME, "arguments": {RESPOND_ACTION_FIELD_NAME: response_state.get("message_flow", "") or response_state.get("response", "")}})
                logger.info("Action spaces: " + json.dumps(action_spaces))
                params_history_str = format_chat_history(params["history"])
                logger.info(f"{params_history_str=}")
                prompt = (
                    sys_instruct + "\n#Available tools\n" + json.dumps(action_spaces) + REACT_INSTRUCTION + "\n\n" + "Conversations:\n" + params_history_str + "Your current task is: " + node_info["attribute"].get("task", "") + "\nThougt:\n"
                )
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": prompt}
                ]


                examiner_turn = message_state.get("examiner_turn", "MISSING")
                #print("This is the EXMAINER_TURN IN orchestrator.py before calling generate_next_step")
                #print(examiner_turn)
                _, action, _ = self.generate_next_step(messages, message_state)
                #print(f"[DEBUG] 📢 Predicted Action: {action}")
                if action is None:
                    #print(f"[DEBUG] ❌ Action is None! Message History: {params['history']}")
                    examiner_turn = message_state.metadata.get("examiner_turn", False)
                    action = "ExaminerWorker" if examiner_turn else "StudentWorker"
                    #print(f"[INFO] 🔄 Assigned Default Action: {action}")

                logger.info("Predicted action: " + action)
                if action == RESPOND_ACTION_NAME:
                    FINISH = True
                else:
                    if action is None:
                        print(f"[ERROR] ❌ Expected Worker Not Assigned! Using Default.")
                        action = "ExaminerWorker" if message_state.get("examiner_turn", False) else "StudentWorker"
    
                    #print(f"\n[DEBUG] 🛠️ Calling self.env.step() with action: {action}")
                    #print(f"[DEBUG] 📜 Message State Before Step: {message_state}")
                    #print(f"[DEBUG] 🚦 Examiner Turn Before step(): {message_state.get('examiner_turn', 'MISSING')}")
                    #if action not in self.env.name2id:
                        #print(f"[ERROR] ❌ Action {action} not found in name2id mapping! Available: {self.env.name2id.keys()}")
                    
                    #print(f"[DEBUG] 🛠️ Type of message_state: {type(message_state)}")
                    #print(f"[DEBUG] 🛠️ Content of message_state: {message_state}")
                    #print(f"[DEBUG] 📢 Examiner Turn from metadata BEFORE step: {message_state['examiner_turn']}")

                    
                    if "examiner_turn" not in message_state:
                        print(f"[ERROR] ❌ 'examiner_turn' is missing from metadata! Defaulting to False.")
                        message_state["examiner_turn"] = False  # Default to avoid missing keys
                    

                    response_state, params = self.env.step(self.env.name2id[action], message_state, params)
                    #ANOTHER BLOCK OF CODE ADDED:
                    if message_state.get("examiner_turn", False) and response_state.get("response"):
                        print(f"[🛑 EARLY RETURN] Examiner Question = {response_state['response']}")
                        return {
                            "answer": response_state["response"],
                            "tool_response": response_state,
                            "parameters": params
                        }

                    #tool_response = params.get("metadata", {}).get("tool_response", {})            #THIS HAS JUST BEEN REPLACED
                    tool_response = response_state if response_state.get("response") else {}
                    #print(f"[✅ FIXED] tool_response = {tool_response}")


        if not response_state.get("response", ""):
            logger.info("No response from the ReAct framework, do context generation")
            tool_response = {}
            if stream_type is None:
                response_state = ToolGenerator.context_generate(response_state)
            else:
                response_state = ToolGenerator.stream_context_generate(response_state)

        response = response_state.get("response", "")
        params["metadata"]["tool_response"] = {}
        # TODO: params["metadata"]["worker"] is not serialization, make it empty for now
        params["metadata"]["worker"] = {}
        params["tool_response"] = tool_response
        #output = {
            #"answer": response,
            #"parameters": params
        #}
        # Capture examiner tool response if present
        if tool_response and tool_response.get("response"):
            output = {
                "answer": tool_response["response"],  # Prefer tool response
                "tool_response": tool_response,
                "parameters": params
            }
        else:
            output = {
                "answer": response,  # fallback assistant output
                "tool_response": {},
                "parameters": params
            }


        with ls.trace(name=TraceRunName.OrchestResponse) as rt:
            rt.end(
                outputs={"metadata": params.get("metadata"), **output},
                metadata={"chat_id": metadata.get("chat_id"), "turn_id": metadata.get("turn_id")}
            )
        #print(f"[FINAL RESPONSE] answer = {output['answer']}")
        #print(f"[FINAL RESPONSE] tool_response = {output['tool_response']}")
        #print(f"[LOOP DEBUG] Node ID: {node_info['id']} | Status: {status} | Response: {response_state.get('response', '')}")


        return output
