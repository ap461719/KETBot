
import uuid
from openai import OpenAI
from arklex.env.workers import register_worker
from dotenv import load_dotenv
import random
import traceback
import re

load_dotenv()  
client = OpenAI()

@register_worker
class ExaminerWorker:
    """
    AI Examiner for the KET Exam (Stages 1 and 2).
    - Stage 1: Dynamically generates related but different questions for both students.
    - Stage 2: Provides discussion topics and facilitates conversation.
    """
    id = uuid.uuid4()
    name = "ExaminerWorker"
    description = "Handles examiner role in KET Stages 1 and 2 with dynamic questioning and discussion topics."

    def __init__(self):
     

        # Stage 2 setup
        self.stage_2_discussion_topics = [
            "There are many different places to eat. Talk together about your favorite places to eat and why.",
            "Talk about your favorite hobbies and why you enjoy them.",
            "Discuss different types of holidays you like and why.",
            "Talk about your favorite school subjects and what makes them interesting.",
            "Discuss your favorite sports and why you like them."
        ]

    def introduce_stage_2(self):
        """
        Provides the introductory message for Stage 2.
        """
        return (
            "Now, in this part of the test, you will talk together. "
            "I’ll give you a topic and I’d like you to have a conversation about it."
        )
    
    def extract_question_by_difficulty(self, raw_response: str, difficulty: str) -> str:
        # Match from "**Difficulty:**" to the next "**" (or end of string)
        pattern = rf"\*\*{difficulty}\*\*:\s*(.*?)(?=\s*\*\*|$)"
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            return match.group(1).strip(" :–-\"\n")  # Clean up trailing symbols
        return raw_response.strip()


    def select_stage_2_topic(self):
        """
        Selects a random discussion topic for Stage 2.
        """
        return random.choice(self.stage_2_discussion_topics)

    def generate_stage_1_question(self, previous_student_response, topic, addressed_student, is_first_speaker):
        """
        Generates dynamic Stage 1 questions using GPT-4o.
        - AI student gets a straightforward question.
        - Real student gets a similar but differently worded question.
        """
        #print("THIS IS THE TOPIC FOR STAGE 1 IN GENERATE_STAGE_1_QUESTION")
        #print(topic)
        #print("this is the is_first_speaker variable print")
        #print(is_first_speaker)
        if addressed_student == is_first_speaker:
            prompt = f"""
            You are a KET examiner. Your task is to ask the student a basic question about the topic: {topic}.
            Generate a clear, fixed QUESTION that matches standard KET format.
            You MUST only ask a question
            Do NOT answer it.
            Topic: {topic}.
            This is what you should ask:
            - Topic: name → "What’s your name?"
            - Topic: age → "How old are you?"
            - Topic: location → "Where do you live?"
            - Topic: friends → "Tell me about your best friend."
            """
        else:
            prompt = f"""
            You are a KET examiner conducting a speaking test. 
            Topic: {topic}.
            The AI student just said: "{previous_student_response}"

            Now generate a follow-up question for the real student:
            - Topic: name → "And you, What’s your name?"
            - Topic: age → "And you, How old are you?"
            - Topic: location → "And you, Where do you live?"
            - Topic: friends → "And you, Tell me about your best friend?"
            """

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        )
        return completion.choices[0].message.content.strip().replace('\n', ' ')
    
    def format_conversation_history(self, convo):
        """
        Turn [{'role': 'assistant', 'name': 'ai_student', 'content': 'I like apples'}, ...]
        into:
        AI Student: I like apples
        You: I like oranges
        """
        #print("THIS IS THE CONVO WE THAT HAS BEEN PASSED ON TO STAGE 3")
        #print(convo)

        formatted = ""

        for turn in convo:
            role = turn.get("role", "")
            name = turn.get("name", "")
            content = turn.get("content", "")

            if role == "assistant" and name == "ai_student":
                formatted += f"AI Student: {content}\n"
            elif role == "user":
                formatted += f"You: {content}\n"

        return formatted.strip()
    
    def generate_stage_3_followup_question(self, discussion_topic, conversation_history, difficulty):
        """
        Generate a follow-up question using both the discussion topic and actual Stage 2 conversation.
        """
        formatted_convo = self.format_conversation_history(conversation_history)
        #print("THIS IS THE FORMATTED CONVO FOR STAGE 3")
        #print(formatted_convo)

        prompt = f"""
    You are a KET speaking test examiner. Your Task is to create a Stage 3 Question that follows the following rules:

    The students just discussed the topic:
    "{discussion_topic}"

    Here’s a summary of their conversation:
    {formatted_convo}

    Now ask follow-up questions that:

    - Matches the difficulty level: {difficulty}
    - It should ask specific questions on what the students have said in {formatted_convo}
    - Stays on the same topic
    - Should be a question for both students (starting with the AI student)
    - Contains the topic within the question

    Examples:
    - Easy: "Do you like this {discussion_topic}?"
    - Medium: "Why do you think {discussion_topic} is important to people?"
    - Hard: "Can you describe how {discussion_topic} can affect someone's emotions or mood?"
    """

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        )

        return completion.choices[0].message.content.strip()



    def ask_stage_1_question(self, previous_student_response, addressed_student, current_topic, message_state):
        """
        Generates the next Stage 1 question, alternating between students.
        """
        #print("[DEBUG] 🎤 Asking Stage 1 Question in ExaminerWorker... in ASK STAGE 1 QUESTION")
       
        
        previous_student_response = message_state.get("previous_student_response", "")
        is_first_speaker = message_state.get("is_first_speaker", "")
        raw_question = self.generate_stage_1_question(
            previous_student_response, current_topic, addressed_student, is_first_speaker
        )
       

        #print(f"[DEBUG] ✅ Returning Stage 1 question | Topic: {current_topic} | Addressed Student: {addressed_student}")
        #print("the raw question is ")
        #print(raw_question)

        return {
            "response": raw_question,
            "examiner_question": raw_question,
            "addressed_student": addressed_student,
            "examiner_turn": False  # Student's turn now
            
        }

    def introduce_and_provide_stage_2_topic(self, custom_topic=None):
        """
        Provides introduction and a random discussion topic for Stage 2.
        """
        intro_message = self.introduce_stage_2()
        discussion_topic = custom_topic if custom_topic else "Topic was not chosen"

        return {
            "response": f"{intro_message} {discussion_topic}",
            "discussion_topic": discussion_topic,
            "examiner_turn": False
        }

    def execute(self, message_state):
        """
  
        Handles execution for both stages.
        - Stage 1: Alternating questions to AI and real student.
        - Stage 2: Provides discussion topic and introduction.
        """
        #print("\n[DEBUG]  execute() was called in ExaminerWorker!")
        #traceback.print_stack()

        stage = message_state.get("stage", "stage_1")
        examiner_turn = message_state.get("examiner_turn", False)
        previous_student_response = message_state.get("previous_student_response", "")
        addressed_student = message_state.get("addressed_student", "ai")
        current_topic = message_state.get("current_topic")

        #print(f"[DEBUG]  Stage: {stage}, examiner_turn: {examiner_turn}")

        if stage == "stage_1" and examiner_turn:
            #print("[DEBUG]  Generating Stage 1 question... in EXECUTE")
            question_output = self.ask_stage_1_question(previous_student_response, addressed_student, current_topic, message_state)
            #print("this is question output AFTER RETURNING FROM ASK_STAGE_1")
            #print(question_output)
            return {
                **question_output,  # Spread operator to include all returned fields
                "user_message": previous_student_response or "No response provided"
            }

        elif stage == "stage_2" and examiner_turn:
            #print("[DEBUG]  Executing Stage 2 discussion...")
            chosen_topic = message_state.get("discussion_topic", None)
            #print("This is the discussion topic in examiner_worker.py")
            #print(chosen_topic)
            stage_2_output = self.introduce_and_provide_stage_2_topic(custom_topic=chosen_topic)
            return stage_2_output  # Already structured correctly
        
        elif stage == "stage_3" and examiner_turn:
            #print("[DEBUG] Executing Stage 3 follow-up question generation...")

            discussion_topic = message_state.get("discussion_topic", "a recent topic")
            difficulty = message_state.get("difficulty", "Medium")
            #previous_response = message_state.get("previous_student_response", "")
            conversation_history = message_state.get("stage_2_conversation", "")
            #print("this is the stage 2 conversation history that is being printed in execute")
            #print(conversation_history)
            addressed_student = message_state.get("addressed_student", "ai")

            follow_up_question = self.generate_stage_3_followup_question(discussion_topic, conversation_history, difficulty)

            # Use transition only on the first subtopic phase of Stage 3
            #if not hasattr(self, "stage_3_subtopic_phase"):
                #self.stage_3_subtopic_phase = 0

            #if self.stage_3_subtopic_phase == 0:
                #transition_text = (
                #"Thank you both for sharing your ideas. Now, let's continue with some questions about your conversation topic."
                #)
                #full_response = f"{transition_text} {follow_up_question}"
            #else:
                #full_response = follow_up_question

            # Alternate turns between AI and Real student like Stage 1
            #addressed_student = "real" if self.stage_3_subtopic_phase == 0 else "ai"
            #self.stage_3_subtopic_phase = (self.stage_3_subtopic_phase + 1) % 2

            return {
                "response": follow_up_question,
                "examiner_question": follow_up_question,
                "addressed_student": addressed_student,
                "examiner_turn": False,
                "stage": "stage_3",
                "discussion_topic": discussion_topic
            }

        #print("[DEBUG]  ExaminerWorker did not recognize the stage. Check if it's set correctly.")
        return {
            "response": "",
            "examiner_question": "",
            "addressed_student": "",
            "user_message": previous_student_response or "No response provided"
        }

