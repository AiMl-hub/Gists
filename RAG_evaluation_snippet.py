import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
from tqdm import tqdm

import google.generativeai as genai

os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize the Generative Model
model = genai.GenerativeModel("gemini-1.5-flash")


def evaluate_answer(prompt):
    """Run the evaluation prompt and return the user's response."""
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=1,
            # stop_sequences=["x"],
            max_output_tokens=20,
            temperature=1.0,
        ),
    )

    return response.text


# Define the evaluation prompts
correctness_prompt_template = """
I am evaluating the factual accuracy of a generated answer. 

**Generated Answer:** {generated_answer}
**Expected Answer:** {ground_truth}

Based on the expected answer, is the generated answer factually correct? 
Please answer with "1" for correct and "0" for incorrect.
"""

coherence_prompt_template = """
I am evaluating the coherence between a generated answer and a question. 

**Question:** {question}
**Generated Answer:** {generated_answer}

Does the generated answer make sense in relation to the question? Does it provide a relevant and coherent response? 
Please answer with "1" for coherent and "0" for incoherent.
"""

relevance_prompt_template = """
I am evaluating the relevance of a generated answer to a retrieved context. 

**Question:** {question}
**Retrieved Context:** {context}
**Generated Answer:** {generated_answer}

Does the generated answer directly address the question and draw relevant information from the provided context? 
Please answer with "1" for relevant and "0" for irrelevant.
"""

# Load the data
data = pd.read_csv("llm_answers.csv")

# Generate scores for each row
count = 0
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    correctness_prompt = correctness_prompt_template.format(
        ground_truth=row["Ground Truth"], generated_answer=row["Generated Answer"]
    )
    print(correctness_prompt)

    coherence_prompt = coherence_prompt_template.format(
        question=row["Question"], generated_answer=row["Generated Answer"]
    )
    print(coherence_prompt)

    relevance_prompt = relevance_prompt_template.format(
        question=row["Question"],
        context=row["Retrieved Context"],
        generated_answer=row["Generated Answer"],
    )
    print(relevance_prompt)

    correctness_score = evaluate_answer(correctness_prompt)
    coherence_score = evaluate_answer(coherence_prompt)
    relevance_score = evaluate_answer(relevance_prompt)
    print("\nCorrectness Score: ", correctness_score)
    print("\nCoherence Score: ", coherence_score)
    print("\nRelevance Score: ", relevance_score)
    data.at[index, "Correctness Score"] = correctness_score
    data.at[index, "Coherence Score"] = coherence_score
    data.at[index, "Relevance Score"] = relevance_score
    data.at[index, "Best Score"] = 1
    count += 1
    if count % 5 == 0:
        time.sleep(60)  # Sleep for 1 minute to avoid quota limit error


# Evaluate the scores
y_true = data["Best Score"].astype(int)
correctness_y_pred = data["Correctness Score"].astype(int)
coherence_y_pred = data["Coherence Score"].astype(int)
relevance_y_pred = data["Relevance Score"].astype(int)

correctness_accuracy = accuracy_score(y_true, correctness_y_pred)
correctness_precision = precision_score(y_true, correctness_y_pred)
correctness_recall = recall_score(y_true, correctness_y_pred)
correctness_f1 = f1_score(y_true, correctness_y_pred)

coherence_accuracy = accuracy_score(y_true, coherence_y_pred)
coherence_precision = precision_score(y_true, coherence_y_pred)
coherence_recall = recall_score(y_true, coherence_y_pred)
coherence_f1 = f1_score(y_true, coherence_y_pred)

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
    "Relevance Accuracy": relevance_accuracy,
    "Relevance Precision": relevance_precision,
    "Relevance Recall": relevance_recall,
    "Relevance F1-Score": relevance_f1,
}

print("\nFinal Scores: ", scores)
# Convert scores dictionary to JSON
scores_json = json.dumps(scores)

# Save scores to a JSON file
with open("llm_answers_scores.json", "w") as file:
    file.write(scores_json)

# Save scores to a CSV file
data.to_csv(
    "llm_answers_scores.csv",
    index=False,
)
