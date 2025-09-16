import os
import json
import argparse
from openai import OpenAI

from arklex.evaluation.simulate_first_pass_convos import simulate_conversations
from arklex.evaluation.extract_conversation_info import extract_task_completion_metrics
from arklex.evaluation.simulate_second_pass_convos import get_labeled_convos
from arklex.utils.model_config import MODEL

# KET Grading Prompt
KET_GRADING_PROMPT = """You are an examiner grading a KET speaking test.
Grade the student's response based on the KET speaking rubric (Bands 1-5). 

Use the following criteria:
1. **Grammar & Vocabulary**: 
   - Band 5: Good control of simple grammar, appropriate vocabulary.
   - Band 3: Sufficient control of grammar, simple vocabulary.
   - Band 1: Limited grammar control, only isolated words.

2. **Pronunciation**:
   - Band 5: Mostly intelligible, some phonological control.
   - Band 3: Mostly intelligible but limited phonological control.
   - Band 1: Often unintelligible.

3. **Interactive Communication**:
   - Band 5: Maintains conversation with little prompting.
   - Band 3: Needs some prompting and support.
   - Band 1: Requires constant prompting, difficulty maintaining conversation.

**Evaluate the response and return a structured JSON output:**
{
    "speaker": "real" or "ai",
    "grammar_vocabulary": 5,
    "pronunciation": 4,
    "interactive_communication": 5,
    "overall_score": 4.7
}
"""
def load_conversation(convo_file):
    """Load the saved conversation file"""
    with open(convo_file, 'r') as f:
        return json.load(f)
    
def evaluate_response(student_response, speaker_type):
    """
    Evaluates a student's response using the KET rubric and returns structured scores.
    """
    client = OpenAI()

    messages = [
        {"role": "system", "content": KET_GRADING_PROMPT},
        {"role": "user", "content": f"Speaker: {speaker_type}. Response: {student_response}"}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return json.loads(completion.choices[0].message.content)

'''
def evaluate(config):
    task = config['task']
    model_api = config['model_api']
    model_params = config['model_params']
    synthetic_data_params = config['synthetic_data_params']
    
    # First Pass: Simulate Conversations
    print("DEBUG: Checking document paths...")
    for doc in config['rag_docs']:
        print("Expected file:", doc['source'], "Exists?", os.path.exists(doc['source']))

    first_pass_data, goals = simulate_conversations(model_api, model_params, synthetic_data_params, config)

    # Extract items from first pass data
    bot_goal = config.get('builder_objective', None)
    bot_goal = None if bot_goal == "" else bot_goal
    goal_metrics = extract_task_completion_metrics(first_pass_data, bot_goal)

    # Apply KET Evaluation on responses
    ket_scores = []
    for convo in first_pass_data:
        for turn in convo["turns"]:
            speaker = turn["speaker"]  # "real" or "ai"
            response = turn["response"]
            if response.strip():
                graded_score = evaluate_response(response, speaker)
                ket_scores.append(graded_score)

    # Second Pass
    if task == 'all':
        labeled_convos = get_labeled_convos(first_pass_data, model_api, synthetic_data_params, model_params, config)
    else:
        labeled_convos = []

    return first_pass_data, labeled_convos, goal_metrics, goals, ket_scores
'''

def evaluate(config):
    conversation_data = load_conversation(os.path.join(config["documents_dir"], "conversation.json"))


    ket_scores = []
    for turn in conversation_data["turns"]:
        speaker = turn["speaker"]
        response = turn["response"]

        if response.strip():
            graded_score = evaluate_response(response, speaker)
            ket_scores.append(graded_score)

    with open(os.path.join(config.get("output-dir", args.output_dir), "ket_scores.json"), "w") as f:
        json.dump(ket_scores, f, indent=4)
    
    print("Evaluation complete. Scores saved in ket_scores.json.")

    return ket_scores




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_api', type=str)
    parser.add_argument('--model_params', type=dict, default={})
    parser.add_argument('--num_convos', type=int, default=5)
    parser.add_argument('--num_goals', type=int, default=5)
    parser.add_argument('--max_turns', type=int, default=5)
    parser.add_argument('--documents_dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    parser.add_argument('--testset', type=str, default=None)
    parser.add_argument('--task', type=str, default='first_pass', choices=['first_pass', 'all'])
    args = parser.parse_args()

    MODEL["model_type_or_path"] = args.model

    assert args.model_api is not None, "Model api must be provided"
    assert args.config is not None, "Config file must be provided"
    assert args.documents_dir is not None, "Documents directory must be provided"
    if not args.output_dir:
        args.output_dir = args.documents_dir
    
    if not os.path.exists(os.path.join(args.output_dir, 'eval')):
        os.makedirs(os.path.join(args.output_dir, 'eval'), exist_ok=True)

    config = json.load(open(args.config))
    if args.testset:
        testset = json.load(open(args.testset))
    else:
        testset = {}

    config['model_api'] = args.model_api
    config['documents_dir'] = args.documents_dir
    config['model_params'] = args.model_params
    config['synthetic_data_params'] = {'num_convos': args.num_convos, 'num_goals': args.num_goals, 
                                       'max_turns': args.max_turns, 'goals': testset}
    config['task'] = args.task

    first_pass_data, final_convos, goal_metrics, goals, ket_scores = evaluate(config)

    with open(os.path.join(args.output_dir, 'eval', 'goals.json'), 'w') as f:
        json.dump(goals, f, indent=4)

    with open(os.path.join(args.output_dir, 'eval', 'simulate_data.json'), 'w') as f:
        json.dump(first_pass_data, f, indent=4)

    with open(os.path.join(args.output_dir, 'eval', 'labeled_data.json'), 'w') as f:
        json.dump(final_convos, f, indent=4)
    
    with open(os.path.join(args.output_dir, 'eval', 'goal_completion.json'), 'w') as f:
        json.dump(goal_metrics, f, indent=4)

    with open(os.path.join(args.output_dir, 'eval', 'ket_scores.json'), 'w') as f:
        json.dump(ket_scores, f, indent=4)
