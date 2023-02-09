## GPT-J Model Tuning

 We finetuned GPT-J for title and teaser snippet generation.
 
 We evaluated the following approaches for model tuning:
 
 - Classic model [fine tuning](https://github.com/snipaid-nlg/model-tuning/blob/main/GPT-J-6B-8bit-HeadlineGeneration.ipynb)
 - Multitask fine tuning
 - [Prompt tuning](https://github.com/snipaid-nlg/model-tuning/blob/main/GPT-J-6B-8bit-HeadlineGeneration.ipynb)
 
 ### Results
 
Prompt tuning did not prove to be successfull.  
Fine tuning and multitask fine tuning delivered promising results.

### Finetuned models

We finetuned [GPT-J-6B-8bit](https://huggingface.co/hivemind/gpt-j-6B-8bit) for title and teaser generation with multitask finetuning.  
We finetuned on two datasets with different sizes.

| Model | Capabilities | Dataset |
|:------|:--------|:-------------|
| [gptj-title-teaser-1k](https://huggingface.co/snipaid/gptj-title-teaser-1k) | title and teaser generation | 1.000 german online news from varying publishers |
| [gptj-title-teaser-10k](https://huggingface.co/snipaid/gptj-title-teaser-10k) | title and teaser generation | 10.000 german online news from varying publishers |
