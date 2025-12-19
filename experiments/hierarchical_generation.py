from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from pathlib import Path
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16
).to(device)

ROOT = Path(__file__).resolve().parents[0]
DATA_PATH = ROOT / "data" / "dialogue_output_test.json"
OUT_PATH = ROOT / "outputs" / "workspace_hierarchical_results_identify_extract.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 데이터 로드
with DATA_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = [item for i, item in enumerate(data, start=1) if not (1 <= i <= 12)]

# 정보 식별 함수 (1단계)
def identify_required_info(user_input: str) -> str:
    prompt = (
        "다음 사용자 발화를 바탕으로, 적절한 응답을 생성하기 위해 어떤 정보가 필요한지 판단하세요.\n"
        f"사용자 발화: {user_input}\n\n"
        "→ 필요한 정보 종류 또는 조건을 명확하게 나열하세요:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**input_ids, max_new_tokens=128, temperature=0.7)
    return tokenizer.decode(
        outputs[0][input_ids["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

# 정보 추출 함수 (2단계)
def extract_info_from_history(identified_info: str, memory_context: str) -> str:
    prompt = (
        f"이전 대화 내용:\n{memory_context}\n\n"
        "위 대화에서 다음 정보 요소들을 찾아 정리하세요. 만약 해당 정보가 없다면 '없음'이라고 명시하세요:\n"
        f"{identified_info}\n\n"
        "→ 추출된 정보:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**input_ids, max_new_tokens=128, temperature=0.7)
    return tokenizer.decode(
        outputs[0][input_ids["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

# 발화 생성 함수 (3단계)
def generate_response(extracted_info: str, user_input: str) -> str:
    prompt = (
        "다음 정보들을 참고하여, 사용자 질문의 의도를 고려한 자연스러운 응답을 생성하세요.\n\n"
        f"[사용자 질문]: {user_input}\n\n"
        f"[추출된 정보]:\n{extracted_info}\n\n"
        "→ 모델 응답:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**input_ids, max_new_tokens=256, temperature=0.7)
    return tokenizer.decode(
        outputs[0][input_ids["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

# 설정
context_window = 5
conversation_history = []
results = []
turn = 1
i = 0

# 대화 루프
while i < len(filtered_data):
    item = filtered_data[i]
    speaker = item.get("speaker")
    sentence = item.get("sentence")

    if speaker == "a":
        user_input = sentence
        conversation_history.append(f"<start_of_turn>user\n{user_input}<end_of_turn>\n")

        memory_context = "\n".join(conversation_history[-context_window * 2:])

        identified_info = identify_required_info(user_input)
        extracted_info = extract_info_from_history(identified_info, memory_context)
        generated_response = generate_response(extracted_info, user_input)

        if i + 1 < len(filtered_data) and filtered_data[i + 1].get("speaker") == "b":
            ground_truth = filtered_data[i + 1]["sentence"]
            conversation_history.append(f"<start_of_turn>model\n{ground_truth}<end_of_turn>\n")
            i += 2
        else:
            ground_truth = None
            conversation_history.append(f"<start_of_turn>model\n{generated_response}<end_of_turn>\n")
            i += 1

        results.append({
            "turn": turn,
            "user_input": user_input,
            "identified_info": identified_info,
            "extracted_info": extracted_info,
            "generated_output": generated_response,
            "correct_response": ground_truth
        })

        print(f"\nTurn {turn}")
        print(f"User: {user_input}")
        print(f"Identified Info: {identified_info}")
        print(f"Extracted Info from History: {extracted_info}")
        print(f"Model Response: {generated_response}")
        if ground_truth:
            print(f"Ground Truth: {ground_truth}")

        turn += 1

    elif speaker == "b":
        conversation_history.append(f"<start_of_turn>model\n{sentence}<end_of_turn>\n")
        i += 1
        turn += 1
    else:
        conversation_history.append(f"<start_of_turn>{speaker}\n{sentence}<end_of_turn>\n")
        i += 1
        turn += 1

# 결과 저장
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved results to: {OUT_PATH}")

