
This repo is an experimental preference fine-tuning of Qwen2-1.5B model for summarization task

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
