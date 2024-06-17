
This repo is an experimental preference fine-tuning of Qwen2-1.5B model for summarization task

The goal is to re-implement Apple work on training specific LoRA's on top of small LM to perform specific tasks, for example summarization.


### Method

Dataset generated using samples from [RedPajamaV2 dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T), specifically Arxiv, Wikipedia, StackExchange documents.
I have downloaded 1% of data and filtered samples that are longer than 900 and shorter than 1800 tokens.

For test set, I saved documents between 16000 and 100000 characters.

Then, I have used [Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) hosted on [Together.ai](https://www.together.ai/) to generate synthetic preference dataset suited for DPO training. Total dataset costed me about $50.
This model accepts maximum 32k tokens, so only such outputs where inputs are shorter than 32k tokens are saved. 

- Task for LLM is to summarize following extract of text, then provide 2 different summaries, second one being a bit shorter, provide explanations, provide rating of summaries, and return Best Summary. 
- I have parsed output from LLM with regexps and filtered responses that have summaries that are too simillar (levenshtein >0.75) and which have low difference of scores <= 1.
- Some of the criteria for my prompt are taken from [GEval suite](https://github.com/microsoft/promptflow/tree/main/examples/flows/evaluation/eval-summarization)

I have generated both train/test datasets using such approach, where test set containts documents that are significantly longer.

I have published datasets on HF hub: Train, Test

Prompt that I used

Due to gpu constraints, I have fine-tuned at 2048 max sequence len. However, Qwen2 uses `rope_base: 1000000` which is suitable for RoPE extension, so I tested resulting models on context of 8-16K tokens.

I have used DPO algorithm with QLoRa-4bit trained for 2 epochs with learning rate 5e-5 in axolotl

Axolotl config


### Metrics

#### BERTScore
|Model name          | Dataset size | Result     |
| ------------------ | ------------ | ---------- |
|Qwen2-1.5B-Instruct | -            | 0.07       |
|Qwen2-1.5B-Summarize| 8000         | **0.14**   |
|Qwen2-1.5B-Summarize| 20500        | In progress|


I have used BERTScore from [official](https://github.com/Tiiiger/bert_score/tree/master) implementation with `microsoft/deberta-xlarge-mnli` model.
Then I sampled 32 inputs from test set (longer sentences to summarize) and generated summaries. I have reference summaries generated from stronger, Qwen2-72B-Instruct model, which I used as targets for metric.


### Usage

You can use latest model on [huggingface](https://huggingface.co/thepowefuldeez/Qwen2-1.5B-Summarize)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("thepowerfuldeez/Qwen2-1.5B-Summarize", 
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             load_in_4bit=True, attn_implementation="flash_attention_2")

text = <YOUR_TEXT>
messages = [
    {"role": "user", "content": text},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt')
new_tokens = model.generate(input_ids, max_new_tokens=1024)[0][len(input_ids[0]):]
summary = tokenizer.decode(new_tokens, skip_special_tokens=True)
```

### Example:
Summarize this paper: https://arxiv.org/abs/2402.12354

Qwen2-1.5B-Instruct:
```
 'The author of the text introduces the concept of "Low Rank Adaptation", '
 'which is originally introduced in [17]. The author shows that LoRA as '
 'originally introduced in [17] leads to suboptimal finetuning of models with '
 'large embeddings (embedding dimensions). The author demonstrates that using '
 'the same learning rate for A and B does not allow efficient feature '
 'learning. The author also shows that this suboptimality of LoRA can be '
 'corrected simply by setting different learning rates for the LoRA adapter '
 'matrices A and B with a well-chosen fixed ratio. The author calls this '
 "proposed algorithm LoRA+. The author's extensive experiments confirm that "
 'LoRA+ improves performance (1% - 2% improvements) and finetuning speed (∼ 2x '
 'SpeedUp), at the same computational cost as LoRA.'
```

Qwen2-1.5B-Summarize (ours):
```
'The objective of this study is to investigate the impact of the learning '
 'rate of the low-rank adaptation (LoRA) between two parameters, A and B, on '
 'the performance and finetuning efficiency of low-rank embeddings (LoRA+) in '
 'large-scale models (LMAs) with large embedding dimensions (DLDs). Our main '
 'focus is to examine the feasibility of using the same learning rate for A '
 'and B in LoRA, without sacrificing efficiency, and to correct this issue by '
 'setting different learning rates for A and B with a chosen fixed ratio. We '
 'conducted extensive experiments, comparing LoRA+ (a modification of LoRA '
 'with different learning rates) with LoRA (a standard LoRA), and found that '
 'LoRA+ outperforms LoRA significantly in both performance and finetuning '
 'speed, up to 2x speed-up over LoRA. These improvements were achieved despite '
 'the computational cost being higher than that of LoRA. Our results also '
 'suggest that LoRA+ could potentially lead to a 1% to 2% improvement in '
 'performance and 2x speed-up in finetuning for certain applications such as '
 'instruction finetuning, even though the full Finetune LoRA remains the '
 'industry-standard method. Overall, our work provides a framework for '
 'researchers to explore the potential of LoRA+ in improving the performance '
 'and finetuning efficiency of large-scale models.\n'
 '\n'
 'The key finding of this study is that LoRA+ can be used to optimize the '
 'performance and finetuning speed of low-rank embeddings with small embedding '
 'dimensions, even under the constraints of large training datasets. This is '
 'achieved by setting different learning rates for A and B, with a well-chosen '
 'fixed ratio. We also discuss the implications of LoRA+ for future research, '
 'highlighting its role in resource-efficient optimization and improving the '
 'scalability of deep learning algorithms. Our findings contribute to the '
 'broader understanding of the trade-offs between different hyperparameters in '
 'deep learning architectures and highlight the importance of choosing '
 'appropriate hyperparameters in resource-efficient optimization techniques. '
 'We conclude with a discussion of the limitations of applying LoRA+ to '
 'real-world scenarios, emphasizing the need for iterative refinement of '
 'hyperparameter choices to ensure optimal performance and efficiency. \n'
 '\n'
 'In conclusion, the study provides a comprehensive framework for researchers '
 'to explore the potential of LoRA+ in optimizing the performance and '
 'finetuning of low-rank embeddings with small embedding dimensions, even '
 'under the constraints of large training datasets. By setting different '
 'learning rates for A and B and selecting a well-chosen fixed ratio, LoRA+ '
 'can potentially lead to substantial improvements in both performance and '
 'finetuning speed, especially for specific applications such as instruction '
 'finetuning. Future research should continue to refine the hyperparameters '
 'and select the best combination of hyperparameters to achieve the desired '
 'outcomes.'
```
