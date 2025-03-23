from datasets import load_dataset
from llm import ai_chat, ai_chat_chain
from sklearn.metrics import classification_report
import re

num_samples = 100
self_consistency_n = 3
tot_n = 3
answer_regex = r"answer:\s*(\d)"

dataset = load_dataset("glue", "mnli", split="validation_matched")
true_labels = [str(dataset["label"][i]) for i in range(num_samples)]

def format_prompt(base_prompt, premise, hypothesis):
    return base_prompt.replace("[premise]", premise).replace("[hypothesis]", hypothesis)

def process_output(llm_output):
    llm_output = llm_output.lower()
    try:
        return re.search(answer_regex, llm_output).group(1)
    except:
        return "-1"

def find_most_common_label(n_predictions):
    common_labels = []
    for i in range(len(n_predictions[0])):
        labels = [n_predictions[n][i] for n in range(len(n_predictions)) if n_predictions[n][i] != "-1"]

        if len(labels) > 0:
            common_labels.append(max(set(labels), key=labels.count))
        else:
            common_labels.append("-1")

    return common_labels 

   
def evaluate(true_labels, predicted_labels):
    return classification_report(true_labels, predicted_labels)


def experiment(base_prompt):
    predicted_labels = [process_output(ai_chat(format_prompt(base_prompt, dataset["premise"][i], dataset["hypothesis"][i]))) for i in range(num_samples)]
    return evaluate(true_labels, predicted_labels)

def experiment_self_consistency(base_prompt):
    n_predictions = []
    for _ in range(self_consistency_n):
        n_predictions.append([process_output(ai_chat(format_prompt(base_prompt, dataset["premise"][i], dataset["hypothesis"][i]))) for i in range(num_samples)])

    majority_vote_labels = find_most_common_label(n_predictions)
    return evaluate(true_labels, majority_vote_labels)

def experiment_tot(base_prompt, next_prompt, final_prompt):
    predicted_labels = []

    for i in range(num_samples):
        prompts = []
        prompts.append(format_prompt(base_prompt, dataset["premise"][i], dataset["hypothesis"][i]))
        for _ in range(tot_n - 1):
            prompts.append(next_prompt)
        prompts.append(final_prompt)

        predicted_labels.append(process_output(ai_chat_chain(prompts)))

    return evaluate(true_labels, predicted_labels)