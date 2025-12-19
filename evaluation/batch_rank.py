
import json
import openai
from openai import OpenAI

client = OpenAI(api_key="xxxx")


# 데이터 로드 함수
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 파일 경로
input_path_dict = {
    "A": DATA_DIR / "workspace_verifi_test_results.json",
    "B": DATA_DIR / "workspace_test_results_reflection.json",
    "C": DATA_DIR / "workspace_self_consistency_test_results.json",
    "D": DATA_DIR / "workspace_hierarchical_results_identify_extract.json",
}

key_map = {
    "A": "generated_output",
    "B": "refined_output",
    "C": "final_answer",
    "D": "generated_output",
}


def build_prompt(user_input, a, b, c, d):
    return f"""You are given a single user utterance and four different responses (A, B, C, D), each generated using a different method.

Please rank the four responses from 1st to 4th place based on how well each response addresses the user’s input.

[User Input]
{user_input}

[A]
{a}

[B]
{b}

[C]
{c}

[D]
{d}
📌 Evaluation Criteria:

Analytical Contextual Relevance and Interpretive Accuracy  
Does the response accurately interpret the user’s statement based solely on the information provided?  
Is the response cautious and precise without making unwarranted assumptions?

Topical Consistency and Logical Coherence  
Does the response stay strictly within the topic without overreaching or introducing unrelated ideas?  
Is the flow of information logical and contextually aligned?

Factual Accuracy and Reliable Information  
Is the information correct, clearly stated, and based on verifiable facts?  
Does the response avoid hallucinations or introducing unsupported details?

Dialogue Continuation and Topic Sustainability 🔶  
Does the response help continue the conversation naturally rather than just giving a one-off fact?  
Even with limited context, does it guide the conversation forward or reflect user interest?

Clarity of Expression and Conversational Naturalness  
Is the tone conversational, clear, and trustworthy?  
Is the wording free from unnecessary formality or awkward phrasing?

📌 Important Notes:  
Responses that make unsupported assumptions, overexplain, or veer off-topic should be ranked lower.  
Even if brief, responses that are cautious, contextually faithful, and precise should be ranked higher.  
Responses that show flexibility and encourage continuation of the conversation without sounding forced or overly confident are preferable.

Output format: Just return a single string like "DABC", no explanation, no markdown.
"""


# 데이터 로딩
data = {k: load_json(v) for k, v in input_path_dict.items()}

# JSONL 생성
jsonl_path = "batch_gpt_ranking.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for i in range(2, len(data["A"])):
        user_input = data["A"][i].get("user_input", "")
        prompt = build_prompt(
            user_input,
            data["A"][i].get(key_map["A"], ""),
            data["B"][i].get(key_map["B"], ""),
            data["C"][i].get(key_map["C"], ""),
            data["D"][i].get(key_map["D"], "")
        )
        payload = {
            "custom_id": f"turn_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a professional dialogue evaluator."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
        }
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# Batch API 실행
with open(jsonl_path, "rb") as f:
    upload = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print("Batch submitted")
    print("Batch ID:", batch.id)
    print("Status:", batch.status)
