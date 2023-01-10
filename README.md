# Transformers--CLIPSeg
This is the code to train CLIPSeg based on hugging face transformers

For the training file, please go to [/examples/pytorch/contrastive-image-text/run_clipseg.py](https://github.com/weimengmeng1999/Transformers--CLIPSeg/blob/main/examples/pytorch/contrastive-image-text/run_clipseg.py)

Also some changes in modeling_Clipseg.py

# CLIPSeg training summary

CLIPSeg is another model that we want to try to leverage the text/visual prompts to help with our instruments segmentation task. The CLIPSeg can be served for: 1) Referring Expression Segmentation; 2) Generalized Zero-Shot Segmentation; 3) One-Shot Semantic Segmentation

Experiment 1: Training CLIPSeg for EndoVis2017 with Text prompt only

Training stage input: 
- Query image (samples in EndoVis2017 training set)
- Text prompt (segmentation class name/ segmentation class description)
Experiment 1.1: Segmentation class name example: ["Bipolar Forceps"]
Experiment 1.2: Segmentation class description example: 
[“Bipolar forceps”, “double-action fine curved jaws”, “horizontal serrations”, “medical grade stainless stell”, “Surgical grade material”, “includes a handle”, “includes a dark or grey plastic like cylindrical shaft”, “includes a complex robotic joint for connecting the jaws/handle to the shaft”]

Testing stage:

- Input: sample in EndoVis2017 testing set; Text prompt
- Output example (binary) for experiment 1.1: doesn’t work ☹ 


![](https://github.com/weimengmeng1999/Transformers--CLIPSeg/blob/main/exp1.1.png)


 
- Output example (binary) for experiment 1.2: works but results are very similar to the pre-trained CLIPSeg



![](https://github.com/weimengmeng1999/Transformers--CLIPSeg/blob/main/exp1.2.png)


 
-	In EndoVis2017 testing set: Experiment 1.2: mean IOU= 79.92%


Experiment 2: Training CLIPSeg for EndoVis2017 with randomly mix text and visual support conditionals

Training stage: 

- Input: 
- Query image (samples in EndoVis2017 training set)
- Text prompt (segmentation class description)
Segmentation class description example is the same as described in experiment 1.2 
-Visual prompt 
Using the visual prompting tips described in the paper, i.e. cropping the image and darkening the background.

![](https://github.com/weimengmeng1999/Transformers--CLIPSeg/blob/main/vp.png)

Testing stage:

- Input: sample in EndoVis2017 testing set; Text prompt

- Output Example:


![](https://github.com/weimengmeng1999/Transformers--CLIPSeg/blob/main/exp2.png)


 
-	In EndoVis2017 testing set: Experiment 1.2: mean IOU= 81.92% (not much improvement)


	
Ongoing Experiment: Fine-tuning CLIP as well as training CLIPSeg decoder
