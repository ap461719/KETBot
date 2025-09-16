# KET/PET Multi-Agent Speaking Exam Simulation

This project simulates the **Cambridge KET (Key English Test)** and **PET (Preliminary English Test)** speaking exams using a multi-agent framework, built on the [Agent-First-Organization](https://github.com/arklexai/Agent-First-Organization) architecture. The system enables interaction between a real student and an AI student, orchestrated by an AI examiner, across three exam stages. It also provides automated evaluation using the official Cambridge rubric, OpenAI models, and custom rule-based logic.

## 💡 Project Objectives

- Simulate realistic speaking practice for KET/PET exams using multi-agent communication.
- Enable participation of a real user as a student alongside an AI student.
- Evaluate student responses using grammar, vocabulary, and communication-based scoring.
- Incorporate flexible difficulty settings, follow-up generation, and dynamic prompts.
- Provide conversation logs and structured scoring for research and feedback.

## ⚙️ Core Features

- **Multi-Agent Simulation**: 
  - `ExaminerWorker` coordinates the conversation across three speaking stages (Introduction, Discussion, Follow-up).
  - `StudentWorker` represents the AI student whose responses are conditioned by difficulty level (`Easy`, `Medium`, `Hard`).
  - Real student responses are integrated in real time alongside the AI student’s replies.

- **Dynamic Prompting & Question Generation**:
  - Stage 1 questions are dynamically generated for PET using the given topic.
  - Stage 2 includes a conversation starter followed by turn-based interaction.
  - Stage 3 follow-up questions are grounded in the Stage 2 conversation using `format_conversation_history()`.

- **Evaluation Pipeline**:
  - Uses OpenAI Embedding API to detect grammar and vocabulary errors based on a custom mistake list.
  - Scores are refined using GPT-4o, following the official Cambridge KET/PET Band 1–5 rubric.
  - Scores are generated per criterion (Grammar/Vocabulary, Pronunciation, Interactive Communication) and averaged.

- **Conversation Logging**:
  - Full conversation (role, speaker, content) is stored in `conversation_history.json` for reproducibility and evaluation.

- **Prompt Engineering**:
  - Prompt templates stored in `prompts.py` files are used to control the behavior of the examiner, AI student, and evaluators.

- **Customizability**:
  - Topic selection is passed via `run.py`.
  - Difficulty selection adjusts AI student's speaking complexity.
  - Stage 1 questions are hardcoded for KET and dynamic for PET.
  - Stage 3 format follows KET: examiner asks → AI student replies → real student replies.

## 🧠 Technologies Used

- **Language Models**: OpenAI GPT-4o (`gpt-4o`) for question generation, AI student responses, and evaluation.
- **Embeddings**: OpenAI Embedding API for grammar/vocabulary mistake detection.
- **Framework**: Built on Agent-First-Organization with modular `workers`, `orchestrator`, and `tools`.
- **Programming Language**: Python 3.11+
- **Storage**: JSON for conversation logs, config settings, and scoring output.

## 🛠️ Getting Started

1. **Environment Setup**
   - Python 3.11+
   - Install dependencies from `requirements.txt`
   - Set up `.env` with your OpenAI API key.

2. **Configure Role & Topic**
   - Define `student_name`, `difficulty`, and `stage1_topic` in `run.py`.

3. **Run the Simulation**
   ```bash
   python run.py
