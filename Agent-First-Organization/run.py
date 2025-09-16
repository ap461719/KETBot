


import os
import json
import time
import logging
import re
import argparse
import traceback
from dotenv import load_dotenv
from pprint import pprint

from arklex.utils.utils import init_logger
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.model_config import MODEL
from arklex.env.env import Env

load_dotenv()

def save_conversation(history, filename):
    """Save chatbot conversation history to a file."""
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)

def pprint_with_color(data, color_code="\033[34m"):  
    print(color_code, end="")  
    pprint(data)
    print("\033[0m", end="")  

def get_api_bot_response(args, history, message_state, params, env):
    """Send user input to the AI examiner and get a response."""
    
    message_state["difficulty"] = params.get("difficulty", "Medium")
    #print(f"[DEBUG] 🔄 Before API Call: examiner_turn={message_state.get('examiner_turn', 'MISSING')}")
 
    data = {
        "text": json.dumps(message_state),
        "chat_history": history,
        "parameters": params,
        "message_flow": message_state.get("message_flow", [])
    }

    orchestrator = AgentOrg(config=os.path.join(args.input_dir, "taskgraph.json"), env=env)
    result = orchestrator.get_response(data)
    #print("This is the output in API BOT RESPONSE")
    #print(result)

    

    return result['answer'], result['parameters']

def extract_name(user_text, params):
    """Extracts name from a user response using regex."""
    if "user_name" in params:
        return params["user_name"]

    name_patterns = [
        r"my name is ([A-Za-z]+(?: [A-Za-z]+)*)",
        r"i am ([A-Za-z]+(?: [A-Za-z]+)*)",
        r"i'm ([A-Za-z]+(?: [A-Za-z]+)*)",
        r"([A-Za-z]+(?: [A-Za-z]+)*)"
    ]

    for pattern in name_patterns:
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            name = match.group(1)
            params["user_name"] = name
            return name

    return "student"  # Default if no name found

def extract_question_by_difficulty(question, difficulty):
    import re
    pattern = rf"\*\*{difficulty}\*\*:\s*(.*?)(?=\*\*|$)"
    match = re.search(pattern, question)
    return match.group(1).strip(" :–-\"\n") if match else question.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="./examples/ket_exam")
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    args = parser.parse_args()

    os.environ["DATA_DIR"] = args.input_dir
    MODEL["model_type_or_path"] = args.model

    # Load configuration
    config = json.load(open(os.path.join(args.input_dir, "taskgraph.json")))
    env = Env(
        tools=config.get("tools", []),
        workers=config.get("workers", []),
        slotsfillapi=config["slotfillapi"]
    )

    history = []
    params = {}
    stage_2_conversation = []
    student_count = 0 




    difficulty_levels = ["Easy", "Medium", "Hard"]
    while True: 
        user_difficulty = input("Choose difficulty level (Easy / Medium / Hard): ").strip().capitalize()
        if user_difficulty in difficulty_levels:
            break 
        print("Invalid input. Please choose Easy, Medium, or Hard.")
    
    while True: 
        first_speaker = input("Who should answer first in each stage? Type 'ai' or 'real': ").strip().lower()
        if first_speaker in ["ai", "real"]:
            break 
        print("Invalid input. Please type 'ai' or 'real")
    
    available_stage_2_topics = [
        "There are many different places to eat. Talk together about your favorite places to eat and why.", 
        "Talk about your favorite hobbies and why you enjoy them.",
        "Discuss different types of holidays you like and why.",
        "Talk about your favorite school subjects and what makes them interesting.",
        "Discuss your favorite sports and why you like them."
    ]

    print("\nAvailable discussion topics for Stage 2:")
    for i, topic in enumerate(available_stage_2_topics):
        print(f"{i + 1}. {topic}")
    
    while True:
        try: 
            topic_choice = int(input("\nChoose a topic number for Stage 2 (1-5): ").strip())
            if 1 <= topic_choice <= len(available_stage_2_topics):
                break
            else:
               print("Invalid input. Please enter a number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    chosen_stage_2_topic = available_stage_2_topics[topic_choice - 1]
    params["chosen_stage_2_topic"] = chosen_stage_2_topic
    
    params["first_speaker"] = first_speaker
    print(f"Answering order set: {first_speaker} answers first.")
    
    params["difficulty"] = user_difficulty
    print(f"Difficulty level set to: {user_difficulty}")

    # Start conversation
    start_message = "Hello and welcome to the KET Exam Chatbot! Let's begin with some questions."
    history.append({"role": "assistant", "name": "examiner", "content": start_message})
    pprint_with_color(f"Examiner: {start_message}")

    previous_examiner_question = None 
    previous_student_response = None
    response_times = [] 

    try:
        # Stage 1: Individual Questions
        topics = ["name", "age", "location", "friends"]
        topic_index = 0 
        student_turn = params.get("first_speaker", "ai")
        is_first_speaker = params.get("first_speaker", "ai")


        while topic_index < len(topics):
            current_topic = topics[topic_index]

            #Examiner asks the question
            examiner_state = {
               "examiner_turn": True,
                "stage": "stage_1",
                "current_topic": current_topic, 
                "addressed_student": student_turn,
                "previous_student_response": previous_student_response or "", 
                "is_first_speaker": is_first_speaker
            }


           
            output_examiner, params = get_api_bot_response(args, history, examiner_state, params, env)
            history.append({"role": "assistant", "name": "examiner", "content": output_examiner})
            pprint_with_color(f"Examiner: {output_examiner}")
        

            previous_examiner_question = output_examiner

            # AI Student or Real Student responds
            if student_turn == "ai":
                student_state = {
                    "stage": "stage_1",
                    "examiner_turn": False,
                    "examiner_question": previous_examiner_question or "Default question",
                    "addressed_student": "ai",
                    "difficulty": params["difficulty"]
                }
                #print(f"[DEBUG] AI Student Turn: {student_state}")

                output_ai_student, params = get_api_bot_response(args, history, student_state, params, env)
                history.append({"role": "assistant", "name": "ai_student", "content": output_ai_student})
                pprint_with_color(f"AI Student: {output_ai_student.strip()}")

                previous_student_response = output_ai_student
                student_turn = "real"
                student_count += 1

            else:
                start_time = time.time()
                user_text = input("You: ").strip()
                end_time = time.time()
                #response_time = round(end_time - start_time, 2)
                #response_times.append(response_time)
                history.append({"role": "user", "content": user_text})
                previous_student_response = user_text
                student_turn = "ai"
                extracted_name = extract_name(user_text, params)
                pprint_with_color(f"Examiner: Thank you, {extracted_name}.")
                student_count += 1
                #examiner_state["examiner_turn"] = True
                #topic_index += 1
            if student_count == 2:
                topic_index += 1
                student_count = 0

        # Stage 2: Discussion
        print("\n--- Moving to Stage 2: Discussion ---")
        params["stage"] = "stage_2"

        #discussion_intro_state = {"examiner_turn": True, "stage": "stage_2"}
        #discussion_intro, params = get_api_bot_response(args, history, discussion_intro_state, params, env)
        #history.append({"role": "assistant", "name": "examiner", "content": discussion_intro})
        #pprint_with_color(f"Examiner: {discussion_intro}")

        discussion_topic_state = {"examiner_turn": True, "stage": "stage_2", "discussion_topic": params.get("chosen_stage_2_topic", "") }
        #print("This is the discussion topic in run.py")
        #print(params.get("chosen_stage_2_topic", ""))
        discussion_topic, params = get_api_bot_response(args, history, discussion_topic_state, params, env)
        history.append({"role": "assistant", "name": "examiner", "content": discussion_topic})
        pprint_with_color(f"Examiner: {discussion_topic}")

        num_discussion_turns = 5
        current_turn = 0
        student_count_stage_2 = 0 
        #student_turn = "ai"
        student_turn = params.get("first_speaker", "ai")

        while current_turn < num_discussion_turns:
            is_last_turn = current_turn == num_discussion_turns - 1

            if student_turn == "ai":
                ai_discussion_state = {
                    "stage": "stage_2",
                    "discussion_topic": discussion_topic,
                    "previous_student_response": previous_student_response,
                    "addressed_student": "ai",
                    "is_last_turn": is_last_turn,
                    "difficulty": params["difficulty"]
                }
                output_ai_student, params = get_api_bot_response(args, history, ai_discussion_state, params, env)
                history.append({"role": "assistant", "name": "ai_student", "content": output_ai_student})
                stage_2_conversation.append({"role": "assistant", "name": "ai_student", "content": output_ai_student})
                pprint_with_color(f"AI Student: {output_ai_student.strip()}")
                previous_student_response = output_ai_student.strip()
                student_turn = "real"
                student_count_stage_2 += 1

            else:
                start_time = time.time()
                user_text = input("You: ").strip()
                end_time = time.time()
                response_time = round(end_time - start_time, 2)
                #response_times.append(response_time)
                history.append({"role": "user", "content": user_text, "response_time_sec": response_time})
                stage_2_conversation.append({"role": "user", "content": user_text})
                previous_student_response = user_text
                student_turn = "ai"
                student_count_stage_2 += 1

            if student_count_stage_2 == 2:
                current_turn += 1
                student_count_stage_2 = 0 

        #print("I am printing stage 2 conversation history right after STAGE 2 ends")
        #print(stage_2_conversation)

        #save_conversation(history, "conversation_history.json")
        #print("\nConversation history saved.")

        print("\n--- Moving to Stage 3: Follow-Up Questions ---")
        params["stage"] = "stage_3"
        num_stage3_cycles = 2
        #student_turn = "ai"
        student_turn = params.get("first_speaker", "ai")
        previous_student_response = None
        student_count_stage_3 = 0
        cycle = 0

        #for cycle in range(num_stage3_cycles):
        while cycle < num_stage3_cycles:
            # Examiner asks a follow-up question based on discussion_topic + stage_2_conversation
            follow_up_question_state = {
                "stage": "stage_3",
                "examiner_turn": True,
                "addressed_student": student_turn, 
                "discussion_topic": discussion_topic,
                "difficulty": params["difficulty"],
                "stage_2_conversation": stage_2_conversation, 
                "previous_student_response": previous_student_response
            }
            output_followup_question, params = get_api_bot_response(args, history, follow_up_question_state, params, env)
            history.append({"role": "assistant", "name": "examiner", "content": output_followup_question})
            pprint_with_color(f"Examiner: {output_followup_question.strip()}")

            previous_examiner_question = output_followup_question

            if student_turn == "ai": 
            # AI Student responds
                student_state = {
                    "stage": "stage_3",
                    "examiner_turn": False,
                    "examiner_question": previous_examiner_question,
                    "addressed_student": "ai",
                    "difficulty": params["difficulty"],
                    "stage_2_conversation": stage_2_conversation
                }
                output_ai_student, params = get_api_bot_response(args, history, student_state, params, env)
                history.append({"role": "assistant", "name": "ai_student", "content": output_ai_student})
                pprint_with_color(f"AI Student: {output_ai_student.strip()}")
                previous_student_response = output_ai_student.strip()
                student_turn = "real"
                student_count_stage_3 += 1

            else:
            # Real Student responds
                start_time = time.time()
                user_text = input("You: ").strip()
                end_time = time.time()
                #time = round(end_time - start_time, 2)
                #response_times.append(response_time)
                history.append({"role": "user", "content": user_text})
                previous_student_response = user_text
                extracted_name = extract_name(user_text, params)
                pprint_with_color(f"Examiner: Thank you, {extracted_name}.")
                student_turn = "ai"
                student_count_stage_3 += 1
            
            print("This is the ")
            print(student_count_stage_3)
            if student_count_stage_3 == 2:
                cycle += 1
                student_count_stage_3 = 0

        save_conversation(history, "conversation_history.json")
        print("\nConversation history saved.")
        from evaluate_ket_transcript import evaluate_ket_grammar

        stage_2_response_times = [
            turn.get("response_time_sec")
            for turn in history
            if turn.get("role") == "user" and "response_time_sec" in turn
        ]

        results = evaluate_ket_grammar(filepath="conversation_history.json", response_times=stage_2_response_times)
        # Print and save results
        print("\n--- KET Evaluation Results ---")
        print(f"Student: {results['student_name']}")
        print(f"Grammar and Vocabulary (Rule-Based): {results['rule_based_score']} ({results['rule_based_error_count']} errors)")
        print(f"Grammar and Vocabulary (GPT): {results['gpt_score']} — {results['gpt_explanation']}")
        print(f"Interactive Communication (Timing-Based): {results['interactive_communication_score']}")
        print(f"\nFinal Average Score: {results['average_score']}/5.0")

        with open("ket_assessment_results.json", "w") as f:
            json.dump(results, f, indent=2)


    except KeyboardInterrupt:
        print("\nConversation ended by user.")
    
