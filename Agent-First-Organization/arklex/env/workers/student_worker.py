
import uuid
from openai import OpenAI
import traceback
from arklex.env.workers import register_worker
import re

client = OpenAI()

@register_worker
class StudentWorker:
    """
    AI Student Worker for the KET Exam.
    - **Stage 1:** Provides concise A2-level responses to examiner questions.
    - **Stage 2:** Engages in discussion with the real student based on a given topic.
    """
    id = uuid.uuid4()
    name = "StudentWorker"
    description = "Acts as an AI student, providing A2-level responses and participating in discussions."

    def __init__(self):
        self.personality = "Beginner English student, friendly and engaging."
    
    def extract_question_by_difficulty(self, question_string: str, difficulty: str) -> str:
        """
        Extracts the relevant question based on difficulty from a multi-difficulty formatted string.
        Example Input:
        '**Easy:** "What is your name?"  **Medium:** "Tell me about your hometown."  **Hard:** "Describe a tradition in your country and explain why it's important."'
        """

        matches = re.findall(r'\*\*(Easy|Medium|Hard):\*\*\s*"([^"]+)"', question_string)
        question_map = {level: question for level, question in matches}
        return question_map.get(difficulty, question_string)

    def is_open_ended_question(self, question):
        open_ended_keywords = [
            "why", "how", "tell me", "describe", "what do you think", 
            "can you explain", "what kind", "what do you like", 
            "where do you live", "what do you do", "your best friend", 
            "hobbies", "favorite", "what do you enjoy"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in open_ended_keywords)



    def generate_response(self, examiner_question, discussion_topic=None, previous_student_response=None, stage="stage_1", is_last_turn=False, difficulty="Medium"):
        """
        Generates a response based on the stage:
        - **Stage 1:** Simple answers to examiner questions.
        - **Stage 2:** Engages in a conversation based on the discussion topic.
        """
        #print(f"\n[DEBUG] 🎭 StudentWorker generating response. Stage: {stage}, Difficulty: {difficulty}")

        question_type = "open-ended" if self.is_open_ended_question(examiner_question) else "factual"
        adjusted_difficulty = difficulty if question_type == "open-ended" else "Easy"
        if stage == "stage_1":
            prompt = f"""

            You are an AI student in the KET exam at an A2 English level. You are not proficient in English. 
            The examiner asked: "{examiner_question}"

            This question is detected as: {question_type}.
            Adjust your response complexity based on {adjusted_difficulty}:
            - Easy: Respond in 3-5 words. Use basic english vocabulary words.
            - Medium: Respond in a full sentence. Use slightly more advanced english vocabulary words 
            - Hard: Respond in 2-3 sentences with reasoning. Use complex vocabulary words and longer sentences. 

            Example:
            - Easy: "I like football."
            - Medium: "I like playing football because it's fun."
            - Hard: "I love playing football because it keeps me fit and lets me spend time with my friends."
            """
        
        elif stage == "stage_2":
           # print("[DEBUG] 🎭 Entered Stage 2 discussion mode.")
            prompt = f"""
            You are an AI student participating in the KET exam discussion.
            Discussion topic: "{discussion_topic}"
            Previous student response: "{previous_student_response or 'None'}"

            - { "- This is the final turn. Do NOT end with a question. End politely and naturally." if is_last_turn else "- You can ask a follow-up question if appropriate." }

            This question is detected as: {question_type}.
            Adjust your response complexity based on difficulty:
            - Easy: Respond in 3-5 words.Use basic english vocabulary words.
            - Medium: Respond in 1-2 sentences. Use slightly more advanced english vocabulary words.
            - Hard: Respond in 3-4 sentences, providing reasoning or examples. Use complex vocabulary words and longer sentences. 

            Example:
            - Easy: "I like watching movies."
            - Medium: "I enjoy watching movies because they are entertaining."
            - Hard: "I love watching movies, especially action films, because they are exciting and keep me engaged."
            """

        else:
            print("[ERROR] ❌ Invalid stage provided.")
            return "Error: Invalid stage provided."
        
       # print(f"[DEBUG] 🛠 Generating AI student response for question: {examiner_question}")
       # print(f"[DEBUG] Cleaned examiner question after difficulty filtering: {clean_question}")


        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        )

        return completion.choices[0].message.content.strip()

    def execute(self, message_state):
        """
        Handles AI student response generation.
        **Expected message_state keys:**
        - "examiner_question": The question asked by the examiner (Stage 1).
        - "discussion_topic": The topic of discussion (Stage 2).
        - "previous_student_response": The previous student's response for context (Stage 2).
        - "addressed_student": "ai" or "real" to determine if the AI student should respond.
        - "stage": "stage_1" or "stage_2".
        """
        
       # print(f"[DEBUG] 🔎 Addressed Student: {message_state.get('addressed_student', 'MISSING')}")

       # print("\n[DEBUG] 🚀 execute() was called in StudentWorker!")
        #traceback.print_stack()

        difficulty = message_state.get("difficulty", "Medium")
        #print("this is the difficulty level in EXECUTE in student_worker.py")
        #print(difficulty)
        stage = message_state.get("stage", "stage_1")
        examiner_question = message_state.get("examiner_question", "")
        #print("this is the examiner question in EXECUTE in student_worker.py")
        #print(examiner_question)
        discussion_topic = message_state.get("discussion_topic", None)
        #print("this is the discussion topic in EXECUTE in student_worker.py")
        #print(discussion_topic)
        previous_student_response = message_state.get("previous_student_response", None)
        addressed_student = message_state.get("addressed_student", "")
        is_last_turn = message_state.get("is_last_turn", False)

       # print(f"[DEBUG] 📌 Stage: {stage}, Addressed Student: {addressed_student}, Examiner Question: {examiner_question}")
        if stage == "stage_3":
            stage = "stage_1"


        # AI student should only respond when addressed
        if addressed_student != "ai":
           # print("[DEBUG] ❌ AI student is NOT addressed. Remaining silent.")
            return {
                "response": "",  # Stay silent if it's not AI's turn
                "stage": stage,
                "examiner_question": examiner_question,
                "discussion_topic": discussion_topic,
                "addressed_student": addressed_student
            }

       # print("[DEBUG] 🎭 AI Student is responding.")
        response = self.generate_response(
            examiner_question=examiner_question,
            discussion_topic=discussion_topic,
            previous_student_response=previous_student_response,
            stage=stage,
            is_last_turn=is_last_turn,
            difficulty=difficulty
        )

        return {
            "response": response,
            "stage": stage,
            "examiner_question": examiner_question,
            "discussion_topic": discussion_topic,
            "addressed_student": addressed_student,
            "ai_student_response": response
        }
