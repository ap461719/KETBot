import json
import re
from openai import OpenAI

client = OpenAI()

# Generalized grammar/vocab error patterns
COMMON_ERRORS = {
    "missing_article": r"\b(?:eat|see|watch|have|buy|read|want)\s+(?!a\b|an\b|the\b)\w+",
    "incorrect_tense": r"\b(I|He|She|We|They)\s+(go|eat|see|come|do|make)\s+(yesterday|last week|ago)\b",
    "wrong_preposition": r"\b(at|on|in)\s+(morning|Monday|the night|home)\b",
    "plural_mistake": r"\b(two|three|many|several|a few)\s+\b\w+[^s.,!?]\b",
    "verb_agreement": r"\b(He|She|It)\s+(do|go|eat|play)\b|\b(I)\s+(is|was)\b|\b(They|We|You)\s+(was)\b",
    "basic_structural_error": r"\b(am|is|are)\s+(likes|go|eat|play|study)\b"
}

def count_grammar_errors(text):
    error_count = 0
    for pattern in COMMON_ERRORS.values():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        error_count += len(matches)
    return error_count

def score_grammar_band(error_count):
    if error_count > 8:
        return 2
    elif error_count > 4:
        return 3
    elif error_count > 2:
        return 3.5
    elif error_count == 2:
        return 4
    elif error_count == 1:
        return 4.5
    else:
        return 5

def score_interactive_communication(response_times):
    if not response_times:
        return None
    avg_time = sum(response_times) / len(response_times)
    if avg_time > 15:
        return 2
    elif avg_time > 10:
        return 3
    elif avg_time > 6:
        return 4
    else:
        return 5

def evaluate_ket_grammar(filepath="conversation_history.json", response_times=None, student_name="Real Student"):
    # Load conversation
    with open(filepath, "r") as f:
        convo = json.load(f)

    # Extract real student responses
    real_student_utterances = [turn["content"] for turn in convo if turn["role"] == "user"]
    student_text = "\n".join(real_student_utterances)

    # Rule-based scoring
    total_errors = sum(count_grammar_errors(utt) for utt in real_student_utterances)
    rule_based_score = score_grammar_band(total_errors)

    # GPT-based scoring
    rubric_prompt = """You are a Cambridge KET examiner. The following are responses from a student in the KET exam. Based only on grammar and vocabulary, use the following rubric to assign a score:
1 = almost incomprehensible
2 = more than 8 grammar errors
3 = 5-7 grammar errors
3.5 = 3-5 grammar errors
4 = 2 grammar errors
4.5 = 1 grammar error
5 = no grammar errors

Return the score in this JSON format:
{
  "student_name": "Real Student",
  "grammar_and_vocabulary_score": <float>,
  "explanation": "<short explanation>"
}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": rubric_prompt},
            {"role": "user", "content": f"Student responses:\n{student_text}"}
        ],
        temperature=0
    )

    gpt_response_text = response.choices[0].message.content.strip()

    # Debugging output
    print("\n🔍 GPT Response:\n", gpt_response_text)

    try:
        gpt_result = json.loads(gpt_response_text)
        gpt_score = gpt_result["grammar_and_vocabulary_score"]
        explanation = gpt_result.get("explanation", "No explanation provided.")
    except Exception as e:
        print("❌ Failed to parse GPT response as JSON.")
        print("⚠️ Error:", str(e))
        print("📄 Raw content:", gpt_response_text)
        gpt_score = rule_based_score
        explanation = "Fallback to rule-based score due to invalid GPT response."

    average_score = round((rule_based_score + gpt_score) / 2, 2)

    # Interactive communication scoring
    interactive_score = score_interactive_communication(response_times)

    return {
        "student_name": student_name,
        "rule_based_error_count": total_errors,
        "rule_based_score": rule_based_score,
        "gpt_score": gpt_score,
        "gpt_explanation": explanation,
        "average_score": average_score,
        "interactive_communication_score": interactive_score
    }
