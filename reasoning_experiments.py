from glue_evaluation import experiment, experiment_self_consistency, experiment_tot

base_prompt = """Does the following premise entail the hypothesis? Output a number meaning
0: Entailment
1: Neutral
2: Contradication

Premise: [premise]
Hypothesis: [hypothesis]

"""
answer_structure = "Answer: 0, 1 or 2"

vanilla_prompt = base_prompt + "Structure your response as " + answer_structure

cot_prompt = base_prompt + "Provide your step by step thought process, then finish your response with " + answer_structure + " indicating your final choice."

tot_prompt = "Now provide another step by step thought process. Feel free to change your final answer if necessary."

tot_final_prompt = "Now based on all your previous reasoning chains, select the label you think is correct. Structure your response as " + answer_structure

vanilla = experiment(vanilla_prompt) 
sc_vanilla = experiment_self_consistency(vanilla_prompt)
cot = experiment(cot_prompt)
sc_cot = experiment_self_consistency(cot_prompt)
tot = experiment_tot(cot_prompt, tot_prompt, tot_final_prompt)

with open("results.txt", "w") as f:
    f.write("Classification reports:\n\n")
    f.write("Vanilla\n" + str(vanilla) + "\n")
    f.write("Self Consistency (Vanilla)\n" + str(sc_vanilla) + "\n")
    f.write("Chain of Thought\n" + str(cot) + "\n")
    f.write("Self Consistency (CoT)\n" + str(sc_cot) + "\n")
    f.write("Tree of Thoughts (CoT)\n" + str(tot) + "\n")