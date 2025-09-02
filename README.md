# UOE: UNLEARNING ONE EXPERT IS ENOUGH FOR MIXTURE-OF-EXPERTS LLMS

## Abstract

Recent advancements in LLMs unlearning have shown remarkable success in removing unwanted data-model influences while preserving the model's utility for legitimate knowledge. Despite these strides, sparse Mixture-of-Experts (MoE) LLMs--a key subset of the LLM family--have remain unexplored in the context of unlearning. As MoE LLMs are celebrated for their exceptional performance, we ask:How can unlearning be performed effectively and efficiently on MoE LLMs? Our pilot study shows that the dynamic routing nature of MoE LLMs introduces unique challenges, leading to excessive forgetting, uncontrolled knowledge erasure and substantial utility drops when existing unlearning methods are applied. To address this, we propose a novel Selected-Expert Unlearning Framework (SEUF). Through expert attribution, unlearning is concentrated on the most actively engaged experts for the specified knowledge. Concurrently, an anchor loss is applied to the router to stabilize the active state of this targeted expert, ensuring focused and controlled unlearning. SEUF is compatible with various standard unlearning algorithms. Extensive experiments demonstrate that SEUF enhances both forget quality up to 5% and model utility by 35% on MoE LLMs across various benchmarks and LLM architectures (compared to standard unlearning algorithms), while only unlearning 0.06% of the model parameters.

## Requirement

Please refer to [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory) to set up environment.

## Quick Start

Conduct experiment on unlearning DeepSeek using GA+UOE with the following code:

```
llamafactory-cli train examples\Deepseek_WMDP\GA_layer16.yaml
```

## Evaluation

[RWKU Github](https://github.com/jinzhuoran/RWKU) provides code for evaluating RWKU Benchmark results.

[lm-eval Github](https://github.com/EleutherAI/lm-evaluation-harness) provides evaluation for MMLU and WMDP.

## License
This repository is licensed under the Apache-2.0 License.