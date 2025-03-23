# Enhancing LLM Reasoning

An experimentation suite, focused on applying different techniques to enhance an LLM's reasoning capabilities.

This was heavily inspired by [Advancing Reasoning in Large Language Models: Promising Methods and Approaches by Avinash Patil](https://arxiv.org/html/2502.03671v1)

The LLM used is **Llama3-7B**, ran via [Ollama](https://ollama.com) and evaluated on the [GLUE MNLI benchmark dataset](https://huggingface.co/datasets/nyu-mll/glue/viewer/mnli) which tasks LLMs to classify whether a premise entails a specified hypothesis.

### Prompting Techniques
- Vanilla: The LLM recieves the premise and hypothesis and is asked to classify with 0 representing entailment, 1 for neutral and 2 for contradication.
- Self Consistency (Vanilla): The LLM is prompted using the Vanilla prompting technique N times, then the most common labels are used as final predictions (majority voting).
- Chain of Thought: The LLM is also asked to output its step by step thought process, in an
- Self Consistency (Chain of Thought): The LLM is prompted using the Chain of Thought prompting technique N times, then the most common labels are used as final predictions.
- Tree of Thoughts: The LLM is asked to produce N different reasoning chains before commiting to a final answer.

### Notes:
Self Consistency Tie Breakers: In the event of a tie, the first element is chosen to be used.  
Self Consistency also discards any invalid labels generated (-1 in the event the LLM does not follow the answer structure format, and thus no answer can be extracted from its response) for majority voting. If all labels used for majority voting are -1s e.g. [-1, -1, -1], then the most common label will be deemed -1 as no valid labels were generated.