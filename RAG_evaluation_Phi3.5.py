import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

import torch

torch.random.manual_seed(0)

# Load the model and tokenizer
model_path = "microsoft/Phi-3.5-mini-instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)


def evaluate_answer(prompt):
    """Generates scores."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": prompt,
        },
    ]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    output = output[0]["generated_text"].strip()
    print("\nOutput: ", output)
    return output


# Define the prompt templates
correctness_prompt_template = """
I am evaluating the factual accuracy of a generated answer.

**Generated Answer:** {generated_answer}
**Expected Answer:** {ground_truth}

**Task:** Based on the expected answer, is the generated answer factually correct?
**Response Format:** Choose one option:
1. **1:** The generated answer is factually correct.
2. **0:** The generated answer is factually incorrect.
"""

coherence_prompt_template = """
I am evaluating the coherence between a generated answer and a question.

**Question:** {question}
**Generated Answer:** {generated_answer}

**Task:** Does the generated answer make sense in relation to the question? Does it provide a coherent response?
**Response Format:** Choose one option:
1. **1:** The generated answer is coherent.
2. **0:** The generated answer is incoherent.
"""

relevance_prompt_template = """
I am evaluating the relevance of a generated answer to a retrieved context.

**Question:** {question}
**Retrieved Context:** {context}
**Generated Answer:** {generated_answer}

**Task:** Determine whether the generated answer directly addresses the question and draw relevant information from the provided context.
**Response Format:** Select one of the following:
1. **1:** The generated answer is relevant.
2. **0:** The generated answer is irrelevant.
"""

# Load the data
data = pd.read_csv(
    "31Aug_llavamed_results/llm_answers.csv"
)

# Evaluate each row in the data
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    question = row["Question"]
    ground_truth = row["Ground Truth"]
    generated_answer = row["Generated Answer"]
    if "Context" in data.columns:
        context = row["Context"]
    else:
        context = None

    correctness_prompt = correctness_prompt_template.format(
        ground_truth=ground_truth, generated_answer=generated_answer
    )
    print(correctness_prompt)
    correctness_score = 1 if "1" in evaluate_answer(correctness_prompt) else 0

    coherence_prompt = coherence_prompt_template.format(
        question=question, generated_answer=generated_answer
    )
    print(coherence_prompt)
    coherence_score = 1 if "1" in evaluate_answer(coherence_prompt) else 0

    if context is not None:
        relevance_prompt = relevance_prompt_template.format(
            question=question,
            context=context,
            generated_answer=generated_answer,
        )
        print(relevance_prompt)
        relevance_score = 1 if "1" in evaluate_answer(relevance_prompt) else 0

    print("\nCorrectness Score: ", correctness_score)
    data.at[index, "Correctness Score"] = correctness_score
    print("\nCoherence Score: ", coherence_score)
    data.at[index, "Coherence Score"] = coherence_score
    if context is not None:
        print("\nRelevance Score: ", relevance_score)
        data.at[index, "Relevance Score"] = relevance_score
    data.at[index, "Best Score"] = 1


# Calculate the final scores
y_true = data["Best Score"].astype(int)
correctness_y_pred = data["Correctness Score"].astype(int)
coherence_y_pred = data["Coherence Score"].astype(int)
if "Relevance Score" in data.columns:
    relevance_y_pred = data["Relevance Score"].astype(int)

correctness_accuracy = accuracy_score(y_true, correctness_y_pred)
correctness_precision = precision_score(y_true, correctness_y_pred)
correctness_recall = recall_score(y_true, correctness_y_pred)
correctness_f1 = f1_score(y_true, correctness_y_pred)

coherence_accuracy = accuracy_score(y_true, coherence_y_pred)
coherence_precision = precision_score(y_true, coherence_y_pred)
coherence_recall = recall_score(y_true, coherence_y_pred)
coherence_f1 = f1_score(y_true, coherence_y_pred)


if "Relevance Score" in data.columns:
    relevance_accuracy = accuracy_score(y_true, relevance_y_pred)
    relevance_precision = precision_score(y_true, relevance_y_pred)
    relevance_recall = recall_score(y_true, relevance_y_pred)
    relevance_f1 = f1_score(y_true, relevance_y_pred)

scores = {
    "Correctness Accuracy": correctness_accuracy,
    "Correctness Precision": correctness_precision,
    "Correctness Recall": correctness_recall,
    "Correctness F1-Score": correctness_f1,
    "Coherence Accuracy": coherence_accuracy,
    "Coherence Precision": coherence_precision,
    "Coherence Recall": coherence_recall,
    "Coherence F1-Score": coherence_f1,
}
if "Relevance Score" in data.columns:
    scores["Relevance Accuracy"] = relevance_accuracy
    scores["Relevance Precision"] = relevance_precision
    scores["Relevance Recall"] = relevance_recall
    scores["Relevance F1-Score"] = relevance_f1

print("\nFinal Scores: ", scores)

# Save scores to a JSON file
scores_json = json.dumps(scores)
with open(
    "llm_answers_scores.json", "w"
) as file:
    file.write(scores_json)

# Save scores to a CSV file
data.to_csv(
    "llm_answers_scores.csv",
    index=False,
)
