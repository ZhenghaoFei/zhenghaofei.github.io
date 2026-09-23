---
title: "Is Visual Fruit Detection Solved by Vision-Language Models, and What Remains? — In the GPT-6 Astra Era and Beyond"
date: 2026-09-16
---

> **TL;DR** — I tested whether frontier vision-language models can perform fruit detection directly from raw images and a written prompt, without task-specific training. On all 200 images of the StrawDI test set, GPT-6 Astra achieved 60.1 segmentation mAP and 86.0 mean instance IoU — exceeding the dataset’s fully supervised Mask R-CNN benchmark in mAP (45.36) and nearly matching its IoU (87.7), despite using no StrawDI training data. The other VLMs we tested were substantially behind, suggesting that this is still a frontier-model capability rather than a general property of VLMs. Astra also showed surprisingly strong performance on small fruit and transferred qualitatively to difficult private scenes, while the same prompt could return masks together with attributes such as redness, occlusion, and graspability. Important limitations remain, especially very small fruit, latency, cost, local deployment, and dependence on a small number of frontier models. Fruit detection is not universally solved, but the starting point has changed dramatically: a strong VLM is now worth testing before collecting and labeling a new task-specific dataset.

Detecting fruit in images has been a long-standing challenge in agricultural computer vision. It is hard not because we cannot achieve high accuracy on a specific dataset, but because real-world scenarios are highly variable. No two orchards look the same, and the same type of fruit can look very different depending on the environment (weather, lighting, occlusion, etc.) and the fruit itself (variety and horticultural practices). Most deep learning-based fruit detection methods rely on large amounts of labelled data to cover the variability (distribution) in the real world. Detection in out-of-distribution (OOD) scenarios often fails. Unfortunately, due to the inherent variability of real-world conditions and data privacy issues, fruit detection always has to deal with OOD scenarios. This means that one often cannot deploy a model trained in one orchard directly to another orchard without fine-tuning it with new data.

My collaborators and I have been working on improving the generalizability of fruit detection models for a while and have tried many methods, including domain adaptation, GANs (generative adversarial networks) [1], and learning from foundation models [2,3]. These methods have shown improvements, but they are still far from solving the problem.

In the meantime, I have always paid attention to vision-language models (VLMs), which are pre-trained on large-scale image–text pairs and have shown impressive generalization ability across many vision tasks. I have been wondering whether VLMs can help with fruit detection and have kept testing them on fruit detection tasks. However, the results were not promising, especially for precise detection.

General-purpose vision-language models have shown impressive visual recognition and reasoning abilities, including the ability to identify fruits across diverse scenes without task-specific training. However, their performance on precise detection has historically been less reliable, as we can see in the experiments below. Several factors may have contributed to this gap. First, there has often been a training-objective mismatch: many VLMs were trained primarily for image–text alignment, captioning, visual question answering, and instruction following rather than exhaustive instance-level localization [4]. Second, strong semantic representations do not necessarily imply equally precise spatial representations; a model may understand what objects are present without representing where every instance is with the same accuracy [4,5]. Third, visual resolution and tokenization may limit small and dense object perception, particularly when many similar objects occupy only a small number of visual tokens [6]. These should be viewed as characteristics of earlier and current VLM designs rather than fundamental limitations, as newer multimodal models such as GPT-6 Astra may substantially change this trade-off.

What motivated me to re-evaluate VLMs for fruit detection was the release of GPT-6 Astra. I had seen surprisingly strong results from Astra on tasks that appear to require substantial visual grounding and spatial understanding—for example, reconstructing and creating complex 3D scenes using tools such as Blender [7], and directly controlling robotic manipulators without task-specific policy training [8,9]. In one recent evaluation, Astra, used directly as a robot policy, even exceeded the reported performance of a specialized VLA baseline [8]. Astra has also set new highs on several visual-spatial and 3D reconstruction evaluations [7]. These results made me wonder whether the capabilities of general-purpose VLMs had reached a point where they could also achieve state-of-the-art performance on fruit detection—or, more broadly, on conventional visual detection tasks.

In this blog, I will share my experience using GPT-6 Astra and other VLMs for fruit detection, examine what has changed compared with earlier generations of VLMs, and explain why the results surprised me so much.


I am particularly impressed by three aspects of what VLMs may offer from this point forward.

1. **Accuracy.**
   Accuracy is the most fundamental metric for visual detection, and GPT-6 Astra shows a substantial improvement over previous VLMs. In our experiments, it detected strawberries with very high precision, even in challenging scenarios involving occlusion, dense clusters, and large variations in fruit size. More interestingly, when we examined the apparent “errors” carefully, many of them turned out to be annotation disagreements rather than clear model failures. In other words, Astra appears to be approaching ground-truth-level detection accuracy in some of our test cases.

2. **Zero-shot generalization.**
   All of our detection experiments were conducted in a zero-shot setting: the model was neither trained nor fine-tuned on the evaluation dataset. Despite this, GPT-6 Astra was able to detect strawberries accurately across diverse scenes. This level of generalization could fundamentally change how fruit detection systems are developed. Instead of collecting and labeling a large dataset for every new orchard, crop variety, camera setup, or environment, users may be able to apply a general-purpose VLM directly to a new detection task with little or no task-specific training.

3. **Near-infinite flexibility.**
   GPT-6 Astra is not limited to returning a bounding box and class label. For each detected fruit, it can also provide a pixel-level outline of the fruit's visible surface and rich semantic information, such as redness, occlusion, calyx visibility, stem visibility, graspability, confidence, and a natural-language description. The output can also be changed simply by modifying the prompt. This makes the detector highly flexible: users can request new attributes, redefine what counts as a target, or adapt the output to different downstream tasks without retraining the model. In this sense, the system begins to look less like a fixed-purpose detector and more like a general visual perception interface.

## A common experiment on a strawberry detection benchmark

We chose the **Strawberry Digital Images dataset (StrawDI)** as a benchmark for our experiments. It is a well-known dataset in the fruit detection community, and it provides a challenging testbed for evaluating the performance of visual detection models. The StrawDI contains photographs collected at commercial plantations in Huelva, Spain. Its annotated subset, **StrawDI_Db1**, contains 3,100 images at 1008 × 756 pixels, split into 2,800 training, 100 validation, and 200 test images, with an instance mask for every strawberry — including unripe, occluded, distant, and partly cropped fruit. [Official dataset description][strawdi]

We evaluated **all 200 test images (1,132 annotated strawberries)** — the same split the dataset's own paper benchmarks on [10], so our numbers can be set next to its supervised specialist directly. No model was trained or fine-tuned on StrawDI: this is a zero-shot evaluation in the operational sense. It is worth noting that because StrawDI is public, it does not guarantee that the images were absent from model pretraining.

### The task and the controls

Every model received the same unannotated image and the same prompt: to detect all strawberry instances in the image and return nine fields for each fruit — a bounding box, a polygon outlining the fruit’s visible surface, redness, occlusion, calyx visibility, stem visibility, graspability, confidence, and a free-text description. **The polygons were scored as instance segmentation directly against the raw ground-truth masks** — the StrawDI paper's own benchmark task — and the boxes from the same responses were scored as detection as a cross-check. The other attributes have no labels in this benchmark.

The following models with corresponding reasoning effort (thinking) were evaluated, and we will see that the reasoning effort is not the main factor for performance.

| Model name | Reasoning effort |
| --- | --- |
| Kimi K3 | High |
| GLM-5.3-Flash | Max |
| DeepSeek V4.1-Flash | High |
| GPT-6 Astra | Low |


### The exact prompt

The following prompt was identical across all four models and all 200 StrawDI test images; we reproduce it verbatim, including the whole-fruit box definition and the visible-surface polygon definition discussed above.

```text
You are looking at ONE image: a 1008x756 colour photograph of a strawberry plant.

COORDINATE SYSTEM for every coordinate you report: pixel coordinates, origin (0, 0) at the TOP-LEFT corner of the image, x increasing to the RIGHT (valid 0..1007), y increasing DOWNWARD (valid 0..755). Report whole numbers of pixels.

TASK
Produce a complete inventory of the strawberries in this image. For each one, describe it fully and trace the outline of its visible surface:
- "bbox": [x1, y1, x2, y2] - the bounding box of the WHOLE fruit. For a partly hidden fruit, give the box the fruit would occupy if the occluder were not there, not just the visible sliver. Box ONLY the fruit body: do not extend the box to cover the calyx (the green sepal leaves) or the stem when they sit apart from the fruit - a calyx that lies flat against the fruit is of course inside the box.
- "polygon": an ordered list of [x, y] vertices tracing the outline of the fruit's VISIBLE surface - the part you can actually see. Walk the boundary once, clockwise or counter-clockwise, starting anywhere. Where a leaf, stem or another fruit hides part of this fruit, follow the occluder's edge; do NOT extrapolate the shape you cannot see (unlike bbox, polygon covers ONLY what is visible). Same fruit-body rule as bbox: exclude the calyx and stem where they sit apart from the fruit. Use 8 to 20 vertices for a typical fruit and never more than 32; a simple closed outline (no self-crossing, no repeated closing vertex) drawn with straight segments between vertices. The polygon's extent must agree with the visible part of the fruit, and whole-pixel coordinates.
- "redness_pct": 0-100, a CONTINUOUS estimate - do not round to a category. Percent of the fruit's VISIBLE surface that is red, judged by hue and saturation together: 0 = no red at all (green or white), 25 = a pale pink flush, 50 = about half the visible surface is red, 75 = mostly red with pale or green patches left, 100 = fully saturated deep red everywhere. Use the whole range; 63 is a better answer than 60.
- "occlusion_pct": 0-100, also CONTINUOUS. How much of the fruit is hidden behind leaves, stems or other fruit: 0 = fully visible, 25 = a leaf edge clips it, 50 = about half hidden, 75 = mostly hidden, 100 = only a sliver shows.
- "calyx_visible": true if the green calyx (sepal ring) can be seen.
- "peduncle_visible": true if the stem is visible where it meets the fruit.
- "graspable": true if a gripper could pick this fruit right now, judging only from what is visible.
- "confidence_pct": 0-100. How sure you are that this really is a strawberry.
- "description": one or two sentences of plain language describing this fruit: its colour and colour pattern, what is occluding it and where, how big it looks, where it sits on the plant, and anything else a picker would want to know. This is free text - write what you actually see, not a restatement of the numbers.

Report EVERY strawberry you can see, whatever its colour. This is an inventory, not a picking decision: do NOT filter by ripeness and do NOT omit a fruit just because it is green, small, or partly hidden. Partly occluded fruit matters as much as fully visible fruit - include a fruit if you can see any part of it, and say how much is hidden in occlusion_pct. Order the list from largest to smallest apparent size. Do not invent fruit that is not there.

Do not label fruit as 'ripe' or 'unripe' - report the continuous redness and occlusion numbers and describe the fruit in words instead.

Each strawberry must carry EXACTLY the nine fields listed above - no more, no fewer, none renamed. If you have nothing for a field, still include it. Put any extra remarks inside description rather than adding a field.

Reply with ONLY a JSON object and nothing else - no prose, no code fences:
{"strawberries": [{"bbox": [x1, y1, x2, y2], "polygon": [[x, y], [x, y], ...], "redness_pct": 0, "occlusion_pct": 0, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 0, "description": "..."}, ...]}
```

### Metrics

The following metrics are computed for the experiments:

**Intersection over Union (IoU)** measures overlap between a predicted polygon (rasterised) and a ground-truth instance mask. A prediction counts as a true positive at IoU ≥ 0.50; matching is greedy in descending confidence order, each label matched at most once. Unmatched predictions are false positives; unmatched labels are false negatives.

- **Precision** = TP / (TP + FP); **Recall** = TP / (TP + FN); **F1** = 2PR / (P + R).
- **AP50** summarizes the precision–recall curve at IoU 0.50; **mAP50–95** averages AP across IoU thresholds 0.50–0.95 in steps of 0.05. (Note that the VLM outputs do not have the same confidence notation as a trained deep learning detector; the confidence values are textual output from the model)
- **Mean matched mask IoU** is the average IoU of matched prediction–label pairs — the same quantity the StrawDI paper reports as mean per-instance IoU (I²oU).
- **Count MAE** is the mean absolute per-image error in fruit count — inventory error, not localization quality.

Our scorer uses all-points interpolation, not the full COCO protocol's 101 recall thresholds [11]. F1 is our primary metric: the VLMs' reported confidences cluster near 100, so AP ranking is sensitive to ties. 

## Results: instance segmentation on the dataset's own benchmark

Before the numbers, here is what the output actually looks like. GPT-6 Astra's visible-surface polygons on six test scenes — ripe and unripe fruit, dense clusters, and heavy occlusion. **What surprised me most is how close many of these predictions look to ground-truth-quality annotations: when inspecting the apparent “errors” manually, a number of them seem to reflect differences in annotation standards or missed labels rather than clear detection failures. In these examples, the GPT-6 Astra's model output is 'ground truth level'**

![Six StrawDI test scenes with GPT-6 Astra's predicted visible-surface polygons against the ground-truth masks.](/assets/fruit_detection_is_solved_by_vlms/gpt_gallery.jpg)

*Figure 1. GPT-6 Astra, zero-shot, on six StrawDI test scenes. White fill: ground-truth instance masks; green: predictions matched at mask IoU ≥ 0.50; coral: unmatched predictions; MISS: labelled fruit the model never found. Note how the outlines follow occluder edges rather than convex hulls. Scenes selected to show the range of behaviour — including misses (images 2085, 2532, 926) — not a random sample.*

*A note on the figures: the overlays show only the predicted boxes and polygons. The other seven per-fruit fields (redness, occlusion, calyx and stem visibility, graspability, confidence, and the free-text description) are hidden for readability, since they play no role in the scored metrics — a full unedited output example appears in the [demo below](#full-nine-field-output-on-this-image).*

| Model | Precision @50 | Recall @50 | F1 @50 | F1 @75 | AP50 | mAP50–95 | Mean matched mask IoU | Count MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi K3 | 77.7% | 64.8% | 70.7% | 39.9% | 62.0% | 30.9% | 0.75 (median 0.77) | 1.04 |
| GLM | 75.0% | 61.0% | 67.3% | 28.4% | 58.2% | 24.3% | 0.72 (median 0.73) | 1.12 |
| DeepSeek | 73.1% | 59.9% | 65.9% | 26.2% | 55.8% | 22.6% | 0.71 (median 0.72) | 1.11 |
| **GPT-6 Astra** | **91.9%** | **83.8%** | **87.6%** | **75.1%** | **83.5%** | **60.1%** | **0.86** (median 0.89) | **0.75** |

*All metrics at mask IoU; 200/200 images scored for every model.*

GPT-6 Astra matches **948 of 1,132 labelled fruits** (84 unmatched predictions, 184 missed labels), exceeding the runner-up Kimi K3's F1 by **17.0 percentage points**. All four configurations returned valid responses for every test image, so no common-subset correction is needed.

![Segmentation precision, recall, F1, and F1 across IoU thresholds for the four configurations.](/assets/fruit_detection_is_solved_by_vlms/seg_scores.png)

*Figure 2. Left: precision, recall, and F1 at mask IoU ≥ 0.50. Right: F1 as the required IoU tightens — GPT's advantage grows with the localization requirement.*

The same scene makes the gap visible without any numbers — here all four models answer the identical prompt on one crowded test image:

![The four models' segmentation output on the same crowded strawberry scene, StrawDI test image 2532.](/assets/fruit_detection_is_solved_by_vlms/same_scene_seg.jpg)

*Figure 3. StrawDI test image 2532, twenty labelled instances. GPT-6 Astra finds fifteen with one unmatched prediction; the other three models find five to eight and over-segment the cluster. Illustrative example, not a representative sample.*

### Full nine-field output on this image

The figure shows only the geometry. Here is each model's complete output on this same image, rendered as an inventory — one card per reported fruit, with its visible-surface polygon, redness and occlusion, calyx/stem/graspable flags, and confidence (boxes and free-text descriptions omitted for clarity; the full JSON follows below):

![GPT-6 Astra's nine-field output on StrawDI test image 2532, one card per reported fruit.](/assets/fruit_detection_is_solved_by_vlms/demo_2532_gpt.jpg)

<details markdown="1">
<summary><strong>Kimi K3</strong> — 13 fruit reported (6 matched, 7 unmatched)</summary>

![Kimi K3's nine-field output on StrawDI test image 2532.](/assets/fruit_detection_is_solved_by_vlms/demo_2532_kimi.jpg)

</details>

<details markdown="1">
<summary><strong>GLM-5.3-Flash</strong> — 13 fruit reported (5 matched, 8 unmatched)</summary>

![GLM-5.3-Flash's nine-field output on StrawDI test image 2532.](/assets/fruit_detection_is_solved_by_vlms/demo_2532_glm.jpg)

</details>

<details markdown="1">
<summary><strong>DeepSeek V4.1-Flash</strong> — 15 fruit reported (7 matched, 8 unmatched)</summary>

![DeepSeek V4.1-Flash's nine-field output on StrawDI test image 2532.](/assets/fruit_detection_is_solved_by_vlms/demo_2532_deepseek.jpg)

</details>

*The four models' complete responses on StrawDI test image 2532, one card per reported fruit (largest first, as the prompt requests). Cards are rendered from the raw model outputs below, unedited.*

And the same responses verbatim, exactly as returned by each model:

<details markdown="1">
<summary><strong>GPT-6 Astra</strong> — 16 fruit reported (15 matched, 1 unmatched)</summary>

```json
{"bbox": [394.0, 243.0, 501.0, 374.0], "polygon": [[397, 297], [403, 277], [416, 257], [437, 246], [456, 244], [474, 251], [480, 268], [475, 279], [489, 287], [498, 303], [491, 327], [479, 347], [462, 365], [443, 373], [420, 368], [404, 356], [397, 332]], "redness_pct": 61, "occlusion_pct": 8, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Large hanging fruit near the center, cream at the shoulder and orange-red across the lower surface. Green sepals overlap its upper-right edge."}
{"bbox": [337.0, 343.0, 427.0, 465.0], "polygon": [[342, 357], [358, 348], [378, 346], [396, 353], [405, 369], [422, 375], [424, 397], [418, 425], [409, 449], [400, 462], [387, 461], [373, 447], [360, 424], [347, 400], [340, 379]], "redness_pct": 62, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Long pointed fruit hanging below and left of the central fruit. Its upper surface is pale cream, changing to strong red toward the tip; the larger neighboring fruit hides part of its upper-right shoulder."}
{"bbox": [657.0, 142.0, 758.0, 228.0], "polygon": [[661, 165], [673, 151], [688, 144], [704, 145], [721, 154], [737, 170], [747, 188], [755, 201], [755, 214], [744, 223], [724, 227], [703, 224], [682, 219], [666, 208], [658, 191]], "redness_pct": 66, "occlusion_pct": 2, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Broad fruit on the right with a pale upper-left shoulder and a red lower and right surface. It sits apart from the central cluster with its body almost entirely exposed."}
{"bbox": [498.0, 250.0, 581.0, 335.0], "polygon": [[531, 252], [549, 256], [563, 267], [574, 285], [579, 302], [575, 319], [565, 331], [550, 334], [531, 329], [516, 319], [513, 304], [526, 297], [529, 282], [540, 275]], "redness_pct": 72, "occlusion_pct": 24, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 100, "description": "Rounded orange-red fruit immediately right of the largest central strawberry. A prominent green calyx covers its left shoulder and center-left surface, with neighboring fruit close beside it."}
{"bbox": [584.0, 250.0, 661.0, 334.0], "polygon": [[604, 256], [621, 251], [638, 256], [649, 267], [656, 283], [660, 302], [655, 321], [645, 332], [630, 331], [612, 325], [597, 315], [588, 299], [585, 282], [590, 268]], "redness_pct": 0, "occlusion_pct": 7, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Large pale green strawberry at the right edge of the central cluster, with pronounced seeds and a slightly tapered bottom. Sepals and a pale petal overlap the top and left edge."}
{"bbox": [168.0, 35.0, 251.0, 105.0], "polygon": [[184, 48], [204, 43], [219, 36], [228, 44], [244, 48], [247, 60], [239, 78], [227, 99], [219, 104], [216, 86], [202, 78], [181, 71], [174, 61]], "redness_pct": 0, "occlusion_pct": 30, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 99, "description": "Broad light-green fruit in the upper-left background. Leaves and sepals hide its top, and the smaller orange fruit in front obscures its lower-left portion."}
{"bbox": [49.0, 143.0, 122.0, 199.0], "polygon": [[62, 147], [78, 145], [94, 151], [108, 158], [118, 169], [121, 181], [113, 193], [101, 198], [87, 195], [72, 190], [60, 181], [53, 170], [52, 158]], "redness_pct": 0, "occlusion_pct": 8, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 100, "description": "Green fruit at the left side surrounded by white petals and broad sepals. Its exposed body is rounded and densely covered with raised seeds."}
{"bbox": [692.0, 48.0, 768.0, 112.0], "polygon": [[721, 57], [740, 53], [755, 56], [765, 66], [762, 82], [753, 98], [740, 109], [725, 108], [714, 102], [705, 91], [716, 84], [722, 70]], "redness_pct": 0, "occlusion_pct": 26, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 99, "description": "Light-green fruit in the upper-right background above the isolated red fruit. A leaf covers its upper-left surface and a diagonal stem crosses near its shoulder."}
{"bbox": [369.0, 27.0, 434.0, 80.0], "polygon": [[379, 36], [392, 30], [406, 30], [417, 37], [425, 49], [426, 62], [417, 71], [400, 75], [386, 69], [381, 57], [393, 57], [398, 49], [377, 44]], "redness_pct": 0, "occlusion_pct": 38, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 94, "description": "Pale green fruit in the upper background just left of the central stem junction. Several crossing stalks obscure its seeded surface and lower edge."}
{"bbox": [244.0, 50.0, 306.0, 96.0], "polygon": [[247, 57], [259, 51], [273, 53], [285, 59], [296, 68], [301, 80], [294, 88], [280, 92], [266, 87], [256, 79], [251, 68]], "redness_pct": 0, "occlusion_pct": 18, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 99, "description": "Small horizontal green strawberry behind the upper-left cluster. The larger pale fruit overlaps its left edge, and green plant parts cross its right end."}
{"bbox": [146.0, 69.0, 221.0, 124.0], "polygon": [[170, 72], [183, 75], [197, 78], [211, 83], [219, 93], [218, 106], [210, 113], [196, 110], [184, 105], [178, 98], [174, 87]], "redness_pct": 18, "occlusion_pct": 43, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 99, "description": "Small yellow-orange strawberry below the larger green fruit in the upper-left cluster. An upright leaf and white petals conceal much of its left and lower surface."}
{"bbox": [143.0, 118.0, 201.0, 163.0], "polygon": [[156, 121], [168, 122], [179, 126], [192, 133], [198, 140], [192, 150], [181, 159], [169, 161], [158, 154], [150, 146], [146, 135], [148, 126]], "redness_pct": 0, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 100, "description": "Small green strawberry framed by white petals, above and right of the leftmost developed green fruit. It has an exposed pointed right end and a pale green seeded surface."}
{"bbox": [649.0, 333.0, 700.0, 383.0], "polygon": [[660, 338], [674, 336], [686, 342], [693, 352], [699, 368], [695, 378], [685, 381], [672, 376], [660, 367], [653, 355], [652, 345]], "redness_pct": 0, "occlusion_pct": 6, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Small green strawberry dangling below and right of the central cluster. Its oval body is exposed beneath a spreading calyx."}
{"bbox": [396.0, 440.0, 435.0, 486.0], "polygon": [[410, 449], [421, 444], [428, 449], [432, 460], [430, 474], [422, 483], [412, 484], [403, 477], [399, 465], [402, 457]], "redness_pct": 0, "occlusion_pct": 26, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 99, "description": "Small green fruit peeking from behind the bottom tip of the long red strawberry. Its lower body is visible, while the neighboring fruit hides its upper-left portion."}
{"bbox": [846.0, 235.0, 890.0, 272.0], "polygon": [[858, 240], [869, 236], [880, 240], [886, 248], [888, 258], [881, 266], [869, 269], [859, 266], [857, 257], [850, 252]], "redness_pct": 0, "occlusion_pct": 15, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Small green fruit on a long stem at the far right, level with the central cluster's shoulders. Sepals overlap its left edge, but most of the rounded seeded body is visible."}
{"bbox": [844.0, 137.0, 872.0, 165.0], "polygon": [[852, 140], [860, 137], [867, 142], [870, 149], [867, 157], [860, 163], [852, 160], [847, 153], [846, 146]], "redness_pct": 0, "occlusion_pct": 20, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 97, "description": "Tiny dark-green developing strawberry above the other far-right green fruit. Its small body sits within pointed sepals at the end of a curved stalk."}
```

</details>

<details markdown="1">
<summary><strong>Kimi K3</strong> — 13 fruit reported (6 matched, 7 unmatched)</summary>

```json
{"bbox": [408.0, 258.0, 500.0, 375.0], "polygon": [[419, 268], [455, 258], [487, 275], [497, 315], [490, 350], [465, 372], [440, 375], [420, 360], [410, 325], [409, 290]], "redness_pct": 55, "occlusion_pct": 8, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 98, "description": "Large central hanging fruit, white-to-pink on the upper half with the lower half flushed red; green calyx sits on top and a small green sepal overlaps its upper right edge. Nearly fully visible and easy to grasp."}
{"bbox": [345.0, 352.0, 432.0, 462.0], "polygon": [[352, 360], [380, 350], [408, 355], [428, 375], [432, 405], [420, 440], [395, 460], [370, 455], [352, 425], [345, 390]], "redness_pct": 60, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 97, "description": "Tapering fruit hanging at lower-left of the central cluster, deep red on the lower half fading to white-green near the calyx at top. A tiny green berry buds just below its tip."}
{"bbox": [497.0, 252.0, 580.0, 330.0], "polygon": [[505, 265], [535, 250], [565, 258], [580, 280], [575, 310], [550, 328], [520, 325], [500, 305], [497, 280]], "redness_pct": 48, "occlusion_pct": 20, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 95, "description": "Roundish fruit just right of the large central one, orange-red over most of its face with a pale top; its upper left is partly hidden behind a green leafy calyx of a neighbouring fruit."}
{"bbox": [585.0, 245.0, 665.0, 325.0], "polygon": [[592, 258], [620, 245], [648, 252], [663, 275], [660, 305], [640, 323], [610, 320], [590, 300], [585, 275]], "redness_pct": 12, "occlusion_pct": 18, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 96, "description": "Mostly green fruit with pale yellow tones hanging right of the central cluster, its top partly covered by its own large calyx leaves."}
{"bbox": [655.0, 140.0, 758.0, 225.0], "polygon": [[662, 155], [695, 140], [730, 148], [755, 172], [756, 200], [735, 222], [700, 225], [670, 210], [657, 182]], "redness_pct": 58, "occlusion_pct": 10, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 97, "description": "Round fruit resting on the plastic mulch at upper right, red on its right half shading to pale pink-white on the left, calyx visible at its top-left."}
{"bbox": [115.0, 40.0, 240.0, 112.0], "polygon": [[125, 55], [160, 40], [205, 45], [238, 60], [235, 90], [200, 108], [155, 110], [120, 90]], "redness_pct": 22, "occlusion_pct": 30, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 88, "description": "Green fruit at upper-left behind foliage with an orange-red patch on its lower-left face; leaves partly cover it and only part of the body is visible."}
{"bbox": [655.0, 325.0, 705.0, 378.0], "polygon": [[660, 335], [680, 325], [700, 335], [703, 360], [690, 376], [668, 372], [656, 352]], "redness_pct": 8, "occlusion_pct": 22, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 90, "description": "Small green berry hanging below-right of the central cluster on a thin stem, sepals flaring at its top."}
{"bbox": [400.0, 445.0, 442.0, 485.0], "polygon": [[405, 455], [420, 446], [438, 452], [440, 470], [425, 483], [408, 478], [400, 465]], "redness_pct": 3, "occlusion_pct": 15, "calyx_visible": true, "peduncle_visible": false, "graspable": true, "confidence_pct": 85, "description": "Tiny green berry budding just beneath the tip of the red lower-left fruit."}
{"bbox": [690.0, 60.0, 762.0, 105.0], "polygon": [[698, 70], [725, 60], [752, 68], [760, 85], [745, 102], [715, 103], [695, 90]], "redness_pct": 2, "occlusion_pct": 25, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 82, "description": "Small green fruit near the top of the image right of centre, half-tucked under leaves and stems."}
{"bbox": [250.0, 45.0, 330.0, 95.0], "polygon": [[258, 55], [285, 45], [315, 50], [328, 68], [315, 90], [280, 93], [255, 78]], "redness_pct": 0, "occlusion_pct": 30, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 80, "description": "Green berry near the upper-left foliage, partly screened by leaves and stems."}
{"bbox": [350.0, 30.0, 420.0, 68.0], "polygon": [[358, 40], [385, 30], [412, 36], [418, 52], [400, 66], [368, 65], [352, 54]], "redness_pct": 0, "occlusion_pct": 35, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 75, "description": "Small green fruit near the top centre among the leaf canopy, well shaded by surrounding leaves."}
{"bbox": [832.0, 220.0, 892.0, 272.0], "polygon": [[840, 230], [862, 220], [885, 228], [890, 250], [875, 268], [848, 268], [833, 250]], "redness_pct": 5, "occlusion_pct": 25, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 85, "description": "Small green berry dangling from a runner at the right side of the image, sepals visible around it."}
{"bbox": [828.0, 118.0, 868.0, 158.0], "polygon": [[835, 126], [852, 118], [866, 130], [865, 148], [848, 156], [832, 145]], "redness_pct": 0, "occlusion_pct": 30, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 78, "description": "Very small green bud-fruit on the long runner stretching to the upper right."}
```

</details>

<details markdown="1">
<summary><strong>GLM-5.3-Flash</strong> — 13 fruit reported (5 matched, 8 unmatched)</summary>

```json
{"bbox": [396.0, 240.0, 506.0, 366.0], "polygon": [[436, 252], [478, 258], [500, 285], [503, 315], [485, 342], [458, 362], [432, 345], [408, 315], [398, 288], [412, 262]], "redness_pct": 38, "occlusion_pct": 10, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 97, "description": "Large conical berry hanging at the front of the central cluster, pale yellow-white on the upper half and turning salmon-orange-red toward the tip with prominent seeds. Green calyx leaves lie over its shoulder and its stem runs up into the canopy. Fully exposed from the side and below, an easy pick."}
{"bbox": [336.0, 334.0, 431.0, 464.0], "polygon": [[355, 345], [390, 338], [418, 352], [430, 382], [425, 415], [405, 448], [382, 462], [360, 448], [344, 418], [338, 382]], "redness_pct": 30, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 97, "description": "Elongated berry hanging low over the black plastic, creamy white with a green tinge near the calyx and a red-orange flush developing from the tip upward. Sepal leaves partially cover its top edge. Hangs free of foliage on the left side of the cluster."}
{"bbox": [662.0, 148.0, 778.0, 238.0], "polygon": [[672, 168], [700, 152], [730, 155], [755, 175], [770, 200], [762, 225], [735, 235], [705, 230], [682, 212], [668, 190]], "redness_pct": 78, "occlusion_pct": 5, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 98, "description": "Broad, nearly ripe berry at the upper right of the hanging cluster, glossy deep red over most of its surface with a pale greenish-white shoulder where the calyx sits. A leaf edge just touches its upper left. Points down and to the right, well clear of other fruit."}
{"bbox": [585.0, 236.0, 673.0, 325.0], "polygon": [[598, 248], [630, 240], [658, 250], [670, 275], [668, 300], [650, 318], [622, 322], [598, 310], [586, 285], [588, 262]], "redness_pct": 2, "occlusion_pct": 8, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 95, "description": "Plump unripe berry, uniform pale silvery-green with dense embedded seeds, hanging just right of the central orange fruit. Its calyx and a white flower sit directly above it. Almost fully exposed."}
{"bbox": [498.0, 250.0, 590.0, 338.0], "polygon": [[510, 262], [538, 254], [560, 262], [572, 285], [570, 312], [552, 330], [528, 330], [510, 315], [500, 290], [502, 270]], "redness_pct": 60, "occlusion_pct": 40, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 90, "description": "Medium orange-red berry tucked between the big pale berry on its left and the green berry on its right, with green sepal leaves crowding its upper edge. Roughly the right two-fifths of it is hidden, so a gripper would have to reach between neighbours to get it."}
{"bbox": [158.0, 50.0, 242.0, 128.0], "polygon": [[168, 60], [200, 52], [228, 62], [240, 85], [235, 110], [215, 125], [190, 122], [170, 105], [160, 82]], "redness_pct": 15, "occlusion_pct": 30, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 90, "description": "Berry lying flat in the dense top-left foliage, mostly pale green-white with a soft orange blush on its left flank. Petioles and leaflets cross over it, hiding about a third including its stem side; it faces upward into the canopy."}
{"bbox": [705.0, 38.0, 792.0, 102.0], "polygon": [[712, 45], [750, 40], [778, 52], [790, 72], [785, 92], [762, 100], [735, 95], [715, 80], [706, 62]], "redness_pct": 0, "occlusion_pct": 10, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 90, "description": "Green unripe berry hanging on a long bare stem at the top right of the plant, well above the red fruit, light green with visible seeds and its sepal ring at the top. Only a stray leaflet brushes its left side."}
{"bbox": [380.0, 26.0, 460.0, 90.0], "polygon": [[382, 35], [415, 28], [445, 35], [458, 55], [452, 78], [428, 88], [400, 82], [382, 62]], "redness_pct": 0, "occlusion_pct": 25, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 85, "description": "Small green berry near the top centre of the canopy, seen from above through a tangle of petioles; several stems cross it and its sepal leaves blend into the foliage. Not reachable without moving leaves."}
{"bbox": [644.0, 326.0, 707.0, 390.0], "polygon": [[652, 335], [678, 328], [698, 338], [706, 358], [700, 378], [680, 388], [658, 382], [646, 362]], "redness_pct": 0, "occlusion_pct": 5, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 95, "description": "Small bright green berry dangling below the larger silver-green one, hanging free over the plastic with its calyx at the top and a thin stem running up. Fully visible and clean."}
{"bbox": [244.0, 56.0, 310.0, 120.0], "polygon": [[252, 65], [278, 58], [300, 68], [308, 90], [298, 112], [275, 118], [255, 108], [246, 85]], "redness_pct": 0, "occlusion_pct": 20, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 85, "description": "Small green berry beside the orange-blushed one in the top-left foliage, partially crossed by stems and leaf edges on its left. Sits deep in the canopy with white flowers nearby."}
{"bbox": [902.0, 14.0, 967.0, 72.0], "polygon": [[910, 20], [940, 15], [960, 28], [965, 50], [952, 68], [928, 70], [910, 55], [904, 35]], "redness_pct": 0, "occlusion_pct": 25, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 80, "description": "Green berry near the top right corner of the frame, partly behind leaflets and a petiole that cross its upper half. Appears unripe and firmly tucked into the foliage."}
{"bbox": [840.0, 122.0, 894.0, 184.0], "polygon": [[850, 130], [872, 125], [888, 138], [892, 160], [880, 178], [858, 182], [845, 168], [842, 145]], "redness_pct": 0, "occlusion_pct": 15, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 70, "description": "Very small green fruit hanging on a slender stem at mid right, beside a big leaf; it may be a young berry rather than a full-sized one. Sepal leaves wrap its top and a leaf edge shades its right side."}
{"bbox": [386.0, 436.0, 444.0, 492.0], "polygon": [[395, 445], [420, 438], [438, 448], [442, 468], [430, 485], [408, 490], [392, 478], [388, 460]], "redness_pct": 0, "occlusion_pct": 30, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 90, "description": "Tiny green berry hanging directly beneath the white-red berry, its top tucked under that fruit's sepal leaves. Too small and crowded to grip; hangs over the black plastic at the lowest point of the cluster."}
```

</details>

<details markdown="1">
<summary><strong>DeepSeek V4.1-Flash</strong> — 15 fruit reported (7 matched, 8 unmatched)</summary>

```json
{"bbox": [393.0, 240.0, 500.0, 372.0], "polygon": [[398, 262], [418, 246], [448, 241], [476, 247], [492, 264], [499, 296], [495, 330], [481, 355], [459, 370], [435, 371], [413, 360], [400, 341], [394, 305], [394, 276]], "redness_pct": 27, "occlusion_pct": 10, "calyx_visible": true, "peduncle_visible": false, "graspable": true, "confidence_pct": 96, "description": "Large fruit hanging in the middle of the cluster, creamy white to pale pink over most of its surface with a soft pink flush on the left and a deeper pink toward the base. The green calyx lies flat across the top and a neighbouring fruit and leaf edge clip its upper right, but the body hangs clear of the truss."}
{"bbox": [333.0, 325.0, 432.0, 467.0], "polygon": [[345, 352], [364, 337], [389, 334], [411, 343], [424, 363], [426, 394], [420, 423], [408, 447], [389, 462], [367, 464], [350, 452], [339, 431], [334, 401], [334, 373]], "redness_pct": 46, "occlusion_pct": 6, "calyx_visible": true, "peduncle_visible": false, "graspable": true, "confidence_pct": 96, "description": "Long conical fruit hanging at the lower left of the cluster, pale pink and white over the upper half and solid pink-red over the lower half. The leafy calyx sits on top and it hangs almost completely free, with only a small leaf tip touching its upper edge."}
{"bbox": [657.0, 140.0, 758.0, 218.0], "polygon": [[668, 158], [692, 145], [721, 143], [747, 151], [757, 171], [755, 191], [741, 207], [715, 215], [689, 212], [671, 200], [663, 180], [663, 166]], "redness_pct": 63, "occlusion_pct": 10, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 95, "description": "Fruit hanging high on the right, deep red over the lower two thirds with pale pink and cream at the shoulder. A pale calyx and the thin green peduncle are visible where it joins the truss, and a leaf tip grazes the top of the fruit."}
{"bbox": [520.0, 258.0, 605.0, 345.0], "polygon": [[532, 282], [551, 266], [577, 262], [596, 272], [604, 292], [601, 315], [589, 333], [565, 341], [543, 338], [530, 322], [524, 302], [524, 290]], "redness_pct": 54, "occlusion_pct": 18, "calyx_visible": true, "peduncle_visible": false, "graspable": true, "confidence_pct": 92, "description": "Medium fruit pressed against the right side of the big pale one, salmon pink shading to red at the tip with a whitish shoulder. Its left edge merges with the neighbouring fruit and a leaf tip crosses the top, so part of the outline is hidden."}
{"bbox": [588.0, 250.0, 660.0, 338.0], "polygon": [[600, 268], [621, 254], [645, 254], [659, 268], [662, 292], [656, 315], [639, 332], [617, 335], [601, 326], [591, 308], [588, 286], [590, 274]], "redness_pct": 3, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": false, "graspable": true, "confidence_pct": 94, "description": "Nearly full-sized green fruit hanging beside the pink one, uniform green with a slightly lighter, almost white streak down the right side. The calyx lies flat on top and a white petal remnant shows behind it."}
{"bbox": [113.0, 100.0, 188.0, 178.0], "polygon": [[124, 128], [141, 110], [163, 106], [181, 118], [188, 140], [184, 160], [167, 174], [145, 176], [127, 166], [118, 148], [116, 136]], "redness_pct": 2, "occlusion_pct": 20, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 88, "description": "Green berry in the upper-left foliage, still green with a whitish sheen and a white blossom petal pressed against its upper left. Leaves overlap its lower edge, hiding part of the surface."}
{"bbox": [148.0, 60.0, 218.0, 122.0], "polygon": [[160, 84], [176, 68], [198, 64], [213, 76], [218, 96], [211, 112], [193, 120], [171, 116], [158, 104], [152, 92]], "redness_pct": 52, "occlusion_pct": 30, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 82, "description": "Smaller fruit at the top left, pink-red with pale shoulders, sitting among the leaves. Several leaves cut across its lower and right sides, hiding roughly a third of it."}
{"bbox": [383.0, 12.0, 440.0, 72.0], "polygon": [[392, 36], [406, 20], [426, 16], [438, 28], [440, 50], [432, 66], [414, 72], [398, 68], [388, 54], [386, 44]], "redness_pct": 0, "occlusion_pct": 15, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 87, "description": "Small green berry held up above the leaves near the top centre of the plant, fully green with a matte surface. Leaf edges cross its lower left, hiding a little of the outline."}
{"bbox": [248.0, 32.0, 302.0, 92.0], "polygon": [[257, 56], [270, 40], [288, 36], [300, 48], [302, 68], [294, 86], [278, 92], [262, 88], [252, 76], [248, 64]], "redness_pct": 2, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 87, "description": "Small green berry in the upper-left group, green with a pale patch on one side, resting among leaves. Part of its lower edge is behind foliage."}
{"bbox": [688.0, 55.0, 748.0, 108.0], "polygon": [[699, 78], [712, 62], [733, 58], [746, 70], [748, 88], [740, 102], [722, 108], [705, 104], [695, 92], [693, 84]], "redness_pct": 0, "occlusion_pct": 25, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 82, "description": "Tiny green berry tucked among the leaves at the top right, showing as a small green dome with its calyx. Leaves cover its lower left, so close to a quarter of it is hidden."}
{"bbox": [652.0, 322.0, 702.0, 382.0], "polygon": [[662, 344], [674, 330], [690, 328], [700, 340], [702, 358], [694, 372], [678, 380], [664, 374], [656, 360], [654, 350]], "redness_pct": 0, "occlusion_pct": 8, "calyx_visible": true, "peduncle_visible": false, "graspable": true, "confidence_pct": 90, "description": "Small green berry hanging just below and right of the pink cluster, uniformly green with a flat calyx on top. It hangs free with only a leaf shadow crossing its upper edge."}
{"bbox": [828.0, 122.0, 878.0, 178.0], "polygon": [[838, 146], [850, 130], [868, 126], [878, 140], [880, 160], [872, 174], [854, 178], [840, 172], [832, 158], [832, 150]], "redness_pct": 0, "occlusion_pct": 28, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 70, "description": "Very small green fruit at the far right of the upper truss with a white petal-like remnant above it. Leaves hide its left side, leaving only a small green dome visible."}
{"bbox": [0.0, 352.0, 42.0, 418.0], "polygon": [[6, 372], [16, 358], [32, 356], [42, 368], [42, 390], [34, 406], [18, 414], [4, 410], [0, 398], [0, 380]], "redness_pct": 4, "occlusion_pct": 35, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 62, "description": "Small greenish-white berry at the extreme left edge of the frame with a white blossom still attached beside it. The frame edge and leaves hide its left side, so only a narrow part of the fruit is visible."}
{"bbox": [398.0, 440.0, 450.0, 492.0], "polygon": [[408, 458], [420, 444], [436, 444], [446, 458], [448, 476], [440, 488], [424, 492], [410, 486], [402, 472], [400, 464]], "redness_pct": 8, "occlusion_pct": 15, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 88, "description": "Very small berry just below the lower-left pink fruit, pale green with a faint pink tinge and a leaf shadow across it. It sits tucked against the stem and leaf, so part of its edge is hidden."}
{"bbox": [845.0, 235.0, 895.0, 275.0], "polygon": [[853, 252], [864, 240], [880, 240], [890, 250], [891, 264], [882, 273], [866, 275], [855, 268], [849, 260], [848, 254]], "redness_pct": 0, "occlusion_pct": 32, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 65, "description": "Very small green berry on the right end of the upper truss, seen as a small green patch beside the stem. Leaves cover most of it, so only a fraction of the surface shows."}
```

</details>

### Comparison with the dataset's own benchmark

The StrawDI paper's benchmark is exactly this task on exactly this split: Mask R-CNN trained on the 2,800 StrawDI training images reaches mean segmentation AP **45.36** and mean per-instance IoU (I²oU) **87.70** on the 200-image test set. [Pérez-Borrero et al., 2020][paper]

| Method | Task-specific training | Split | Segmentation mAP | Mean instance IoU |
| --- | --- | --- | ---: | ---: |
| Mask R-CNN (original paper) | 2,800 StrawDI images | test (200 images) | 45.36 | 87.70 |
| Kimi K3, zero-shot | none | test (200 images) | 30.9 | 74.9 |
| **GPT-6 Astra, zero-shot** | **none** | **test (200 images)** | **60.1** | **86.0** |
| GPT-6 Astra, zero-shot | none | val (100 images) | 61.3 | 86.9 |

*The paper's numbers are literature reference points, not reruns: the AP protocol differs (all-points interpolation here) and our polygons are limited to 32 vertices. But the task, the output representation, the ground truth, and the test split are the same. The validation row comes from an earlier 100-image batch of the same configuration, included to show the result replicates across splits.*

**A general-purpose VLM with no task-specific training exceeds the fully supervised Mask R-CNN's segmentation mAP on its own test split (60.1 vs 45.36) and lands within two points of its mean instance IoU (86.0 vs 87.7)** — With the validation-split run (61.3 / 86.9) showing the same profile, this is the strongest evidence presented in this article. The other VLMs are still far behind (best non-GPT 6 Astra: Kimi K3 at mAP 30.9 and mean IoU 74.9), suggesting that this level of performance is, for now, a frontier-model capability rather than a general property of VLMs. At the same time, we look forward to seeing whether the next generation of VLMs from other providers, including open-source models, can approach or match this level of performance.

### Small fruit remain challenging — but Astra is already much stronger

Small fruit are still the most difficult category, but GPT-6 Astra performs substantially better than the other VLMs. Its size-stratified mask recall is 338/338 for large fruit, 513/542 for medium fruit, and 97/252 for small fruit (38.5%). While the recall on small fruit is clearly lower than on medium and large fruit, it is still striking compared with the other models, which detect only 7–14 of the 252 small instances.

This suggests that GPT-6 Astra has made a significant step forward even on small-object detection, although very small and weakly visible fruit remain one of the clearest failure modes. In other words, small fruit are no longer simply a case where all VLMs fail similarly — Astra shows a substantial capability gap here as well.

![Recall by visible instance size for the four configurations.](/assets/fruit_detection_is_solved_by_vlms/size_recall.png)

*Figure 4. Recall at mask IoU ≥ 0.50 by visible mask area (small < 1,024 px²; medium 1,024–9,215 px²; large ≥ 9,216 px²). Labels give matched/total instances.*

### An unexpected caveat: silent provider-side routing

One episode during this evaluation is worth reporting because it nearly contaminated the GPT-6 Astra results. Every call was issued with the same recorded model identifier (`gpt-6-astra` via the Codex CLI), and every batch passed the synthetic vision-delivery control. Yet one intermediate retry batch returned responses that were perfectly schema-valid — and visibly wrong on the image: much of the fruit missing, clusters over-segmented, outlines that no longer follow occluder edges. Two of the affected frames, exactly as they came back:

![Silent routing comparison, frame 995: suspect batch vs re-requested.](/assets/fruit_detection_is_solved_by_vlms/routing_995.jpg)

*Figure 5. StrawDI test image 995. Left: the suspect batch's answer — same prompt, same recorded model identifier, schema-valid. Right: the same frame re-requested under identical settings.*

![Silent routing comparison, frame 926: suspect batch vs re-requested.](/assets/fruit_detection_is_solved_by_vlms/routing_926.jpg)

*Figure 6. Test image 926 — the same pattern.*

Re-requesting the affected frames under identical settings immediately recovered the dense, clean inventories shown on the right. Our suspicion is that the provider silently routed those requests to a different, weaker model while the request-side bookkeeping still said `gpt-6-astra`. The affected records were excluded wholesale and re-run; every GPT-6 Astra number in this article comes from the corrected batch. The lesson generalizes beyond this incident: with API-routed frontier models, "the request named the model" is not evidence that the model answered — and no schema validator will catch a wrong-model response.

## Zero-shot evaluation on private data

A public benchmark cannot prove out-of-distribution performance: StrawDI has been downloadable for years, so these images may be in pretraining data. As a check beyond the public benchmark, we ran the identical nine-field census prompt — byte-for-byte the same prompt and schema — on **two private, deliberately chaotic scenes from our own collection**: **private-1**, a dense hanging truss from a greenhouse row, and **private-2**, a hand-held close-up of an overlapping cluster. Neither image has been publicly released, and neither has any ground truth, so this is qualitative: what does the model's inventory look like when the scene is messy and the data cannot have been memorized as a benchmark?

Twelve configurations ran both scenes: GPT-6 Astra (low effort), the six **GPT-5.6** configurations (the `luna` and `sol` variants, each at low, medium, and high reasoning effort), four **GPT-6** `luna`/`sol` configurations (each at low and high effort), and GLM (maximum effort) as a non-GPT reference.

| Configuration | `private-1` | `private-2` |
| --- | ---: | ---: |
| **GPT-6 Astra (low)** | **23** | **21** |
| GPT-6 luna low / high | 20 / 19 | 14 / 16 |
| GPT-6 sol low / high | 22 / 21 | 18 / 17 |
| GPT-5.6 luna low / medium / high | 18 / 21 / 19 | 14 / 16 / 17 |
| GPT-5.6 sol low / medium / high | 17 / 21 / 21 | 18 / 17 / 18 |
| GLM-5.3-flash | 16 | 14 |

*Fruit reported per scene; counts alone say nothing about correctness — the grids below are the evidence.*

The visual verdict is unambiguous, and it indicates two things. First, **GPT-6 Astra is uniquely good**: on both scenes its detections are essentially perfect — almost every fruit I can find by eye is outlined (there are actually some missing but are very small ones hard to tell if they are fruits or not), the outlines closely follow the visible surface, and nothing is invented. Second, **the GPT-5.6 family is fairly flat**: from `luna` medium all the way to `sol` high (`luna` low is apparently the weakest), the five runs produce very similar results — similar counts, similar outlines, and similar misses — so increasing reasoning effort or switching variants does not close the gap to Astra. The four GPT-6 `luna`/`sol` runs land in exactly the same count range as the GPT-5.6 family (19–22 on `private-1`, 14–18 on `private-2`) — so the step up is specific to Astra rather than a property of the 6 generation as a whole, and neither reasoning effort nor the variant switch closes the gap.

**What the `luna` and `sol` runs cost.** Each single-scene call used ≈27k input tokens (≈3.3–3.7k of them cached) and 1.8–4.1k output tokens, and took 57–108 s. The runs record no provider-side cost; at the announced GPT-6 `luna`/`sol` rates ([OpenAI, September 22, 2026](https://openai.com/index/introducing-gpt-6-sol-and-luna/) — `luna`: $0.10 / $0.01 / $0.50 per million input / cached-input / output tokens; `sol`: $2 / $0.20 / $10), that is **≈$0.007–0.008 per scene pair for `luna` and ≈$0.15–0.17 for `sol`** — orders of magnitude below the Astra calls discussed in the cost section below.


![All twelve configurations on private-1, the greenhouse-truss scene.](/assets/fruit_detection_is_solved_by_vlms/chaos_private-1.jpg)

*Figure 7. Scene `private-1` (dense hanging truss). Each polygon is coloured by the model's own reported redness (green → red). GPT-6 Astra (top-left panel) resolves the cluster fruit-by-fruit; the six GPT-5.6 panels and the four GPT-6 `luna`/`sol` panels are nearly interchangeable, and GLM undercounts. Qualitative — no ground truth exists on this scene.*

![All twelve configurations on private-2, the hand-held scene.](/assets/fruit_detection_is_solved_by_vlms/chaos_private-2.jpg)

*Figure 8. Scene `private-2` (hand-held close-up, heavy overlap). Same pattern: GPT-6 Astra's inventory is complete and cleanly separated; the GPT-5.6 and GPT-6 `luna`/`sol` panels again look alike from luna-low to sol-high.*

### Full nine-field output on the private scenes

GPT-6 Astra's outputs on both private scenes, rendered the same way — every reported fruit with its polygon and attributes, including the redness, occlusion and graspability information that the overlay grids above do not show:

![GPT-6 Astra's nine-field output on private-1, the greenhouse-truss scene.](/assets/fruit_detection_is_solved_by_vlms/demo_chaos_private-1.jpg)

![GPT-6 Astra's nine-field output on private-2, the hand-held scene.](/assets/fruit_detection_is_solved_by_vlms/demo_chaos_private-2.jpg)

*GPT-6 Astra's complete inventories of `private-1` (23 fruit) and `private-2` (21 fruit).*

And the complete, unedited responses as returned:

<details markdown="1">
<summary><strong>GPT-6 Astra</strong> on <code>private-1</code> — 23 fruit reported</summary>

```json
{"bbox": [704.0, 414.0, 798.0, 519.0], "polygon": [[732, 419], [750, 416], [767, 422], [782, 440], [786, 457], [795, 476], [795, 494], [786, 509], [772, 518], [756, 510], [738, 498], [718, 487], [708, 472], [706, 453], [715, 435]], "redness_pct": 46, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Large irregular fruit near the lower centre, with a cream-green shoulder and red lower tip. Sepals cover part of its upper surface and a slender stem crosses its right side."}
{"bbox": [1147.0, 394.0, 1245.0, 482.0], "polygon": [[1154, 408], [1168, 400], [1178, 417], [1192, 425], [1207, 411], [1224, 409], [1240, 415], [1243, 434], [1235, 454], [1220, 478], [1209, 482], [1187, 472], [1168, 461], [1154, 444], [1149, 427]], "redness_pct": 73, "occlusion_pct": 18, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Broad red fruit at the far right, with pale peach colouring around its shoulders. A large green calyx spreads over the upper face."}
{"bbox": [1064.0, 363.0, 1148.0, 460.0], "polygon": [[1093, 366], [1110, 367], [1125, 377], [1134, 392], [1136, 410], [1145, 429], [1144, 444], [1131, 454], [1116, 459], [1100, 451], [1081, 447], [1067, 435], [1065, 415], [1071, 393], [1080, 376]], "redness_pct": 94, "occlusion_pct": 13, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Large bright red fruit in the right-hand cluster, with a darker lower end. Thin stems cross its face and neighbouring green fruit crowd its lower edge."}
{"bbox": [391.0, 111.0, 470.0, 208.0], "polygon": [[403, 117], [420, 115], [434, 120], [446, 115], [460, 122], [468, 135], [466, 153], [456, 171], [445, 190], [431, 205], [419, 208], [407, 200], [399, 182], [394, 162], [391, 142], [396, 125]], "redness_pct": 64, "occlusion_pct": 5, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Prominent hanging fruit left of centre, cream-green at the shoulders and increasingly dark red toward its pointed bottom. Its stem is exposed above the small calyx."}
{"bbox": [215.0, 193.0, 293.0, 289.0], "polygon": [[231, 198], [250, 198], [266, 196], [281, 205], [288, 220], [291, 238], [289, 256], [283, 276], [274, 287], [261, 288], [247, 280], [236, 266], [225, 249], [218, 231], [219, 214]], "redness_pct": 48, "occlusion_pct": 9, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Long fruit on the left with green-white colouring on its left shoulder and red colouring along the right side and lower tip. Narrow stems and sepals overlap its top."}
{"bbox": [654.0, 145.0, 730.0, 247.0], "polygon": [[673, 150], [690, 153], [705, 147], [719, 157], [726, 174], [728, 191], [722, 208], [708, 224], [698, 240], [687, 246], [676, 242], [669, 226], [666, 209], [658, 193], [655, 178], [660, 163]], "redness_pct": 39, "occlusion_pct": 7, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Hanging fruit above the centre, pale yellow-green across its broad upper body with pink-red colouring down the left side and narrow lower end. Its supporting stem is visible above."}
{"bbox": [759.0, 550.0, 828.0, 633.0], "polygon": [[772, 556], [787, 559], [802, 554], [817, 560], [826, 573], [825, 591], [818, 609], [807, 626], [797, 632], [786, 628], [776, 614], [767, 596], [760, 577], [763, 565]], "redness_pct": 2, "occlusion_pct": 9, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Pale green heart-shaped fruit hanging lowest in the image. Its seed-covered body is exposed below a spreading calyx, with only a faint warm blush near the top."}
{"bbox": [434.0, 338.0, 496.0, 414.0], "polygon": [[448, 343], [460, 345], [474, 340], [488, 348], [494, 362], [491, 379], [483, 398], [474, 411], [463, 413], [452, 403], [444, 387], [437, 370], [435, 354]], "redness_pct": 0, "occlusion_pct": 8, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Medium pale green fruit in the centre-left group, tapering downward and covered in dark seeds. Its upper edge is partly covered by sepals."}
{"bbox": [549.0, 300.0, 612.0, 366.0], "polygon": [[564, 309], [576, 314], [580, 302], [594, 307], [606, 318], [611, 334], [606, 348], [596, 361], [585, 365], [572, 358], [564, 345], [558, 330], [550, 322]], "redness_pct": 9, "occlusion_pct": 24, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Pale green fruit with a peach tint in the middle cluster. Long sepals and a descending stem cover its upper and left portions, and another small green fruit hangs immediately beneath it."}
{"bbox": [198.0, 92.0, 250.0, 151.0], "polygon": [[211, 97], [225, 95], [237, 100], [246, 111], [246, 125], [239, 137], [226, 148], [215, 145], [207, 134], [200, 120], [199, 109]], "redness_pct": 54, "occlusion_pct": 21, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 98, "description": "Small pink-red fruit at the upper left, with a pale shoulder and darker lower surface. Sepals and crossing stems obscure its top and right edge, beside a darker hanging fruit."}
{"bbox": [219.0, 111.0, 260.0, 180.0], "polygon": [[244, 119], [253, 119], [257, 131], [255, 145], [256, 160], [250, 174], [241, 179], [231, 176], [224, 166], [222, 152], [233, 142], [237, 130]], "redness_pct": 95, "occlusion_pct": 34, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 96, "description": "Narrow dark red fruit behind and below the pink fruit at the upper left. The neighbouring fruit and several diagonal stems obscure much of its upper and left side."}
{"bbox": [733.0, 314.0, 786.0, 362.0], "polygon": [[750, 319], [763, 320], [775, 325], [782, 337], [784, 349], [778, 357], [767, 361], [755, 357], [746, 348], [740, 337], [744, 327]], "redness_pct": 13, "occlusion_pct": 22, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Small yellow-peach fruit just right of centre, pointing down and right. Its upper-left portion sits behind green sepals and an adjacent green fruit."}
{"bbox": [1107.0, 451.0, 1160.0, 513.0], "polygon": [[1121, 458], [1134, 462], [1145, 460], [1154, 473], [1158, 488], [1153, 503], [1144, 511], [1133, 509], [1122, 499], [1115, 485], [1110, 472]], "redness_pct": 0, "occlusion_pct": 19, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Light green fruit at the lower right of the red cluster, angled diagonally downward. Large sepals cover its shoulders and neighbouring fruit crowd its upper-left side."}
{"bbox": [552.0, 355.0, 598.0, 410.0], "polygon": [[563, 365], [574, 370], [585, 363], [594, 369], [596, 381], [590, 395], [580, 408], [572, 410], [563, 402], [557, 389], [554, 376]], "redness_pct": 0, "occlusion_pct": 21, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 99, "description": "Small dark green fruit directly below the pale central fruit. Its calyx and the fruit above hide its shoulders, while its tapered lower body is clearly visible."}
{"bbox": [496.0, 337.0, 548.0, 388.0], "polygon": [[511, 346], [526, 344], [536, 352], [544, 361], [546, 374], [539, 383], [528, 387], [515, 382], [504, 372], [499, 361]], "redness_pct": 0, "occlusion_pct": 22, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Short green fruit between the larger pale green fruit on the left and the central pair. Broad sepals overlap its upper face and left shoulder."}
{"bbox": [707.0, 298.0, 755.0, 354.0], "polygon": [[720, 308], [732, 310], [744, 314], [751, 326], [751, 339], [743, 349], [734, 353], [723, 347], [715, 335], [710, 322]], "redness_pct": 0, "occlusion_pct": 27, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 97, "description": "Small green fruit immediately left of the yellow-peach fruit near the centre-right. Its upper body is covered by a dense calyx, and the neighbouring fruit overlaps its right edge."}
{"bbox": [613.0, 428.0, 662.0, 474.0], "polygon": [[625, 435], [636, 439], [647, 435], [656, 442], [661, 455], [657, 467], [649, 472], [638, 473], [626, 466], [619, 455], [615, 444]], "redness_pct": 0, "occlusion_pct": 13, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Small green fruit suspended just above the horizontal metal rail near the centre. Its rounded, slightly tapered body is mostly exposed beneath a leafy calyx."}
{"bbox": [1056.0, 449.0, 1100.0, 499.0], "polygon": [[1065, 457], [1076, 465], [1086, 457], [1095, 466], [1099, 480], [1093, 492], [1084, 498], [1074, 495], [1064, 486], [1059, 473]], "redness_pct": 0, "occlusion_pct": 22, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Small green fruit beneath the left side of the large red fruit on the right. Long pale sepals drape over its shoulders and its stem is crowded by neighbouring stems."}
{"bbox": [969.0, 434.0, 1006.0, 481.0], "polygon": [[978, 443], [988, 449], [997, 441], [1003, 451], [1003, 463], [997, 476], [988, 480], [980, 474], [975, 463], [972, 453]], "redness_pct": 0, "occlusion_pct": 20, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Small isolated green fruit to the left of the right-hand cluster, hanging in front of the rail. A star-shaped calyx covers its upper body."}
{"bbox": [1030.0, 394.0, 1072.0, 433.0], "polygon": [[1040, 400], [1050, 405], [1061, 405], [1067, 413], [1067, 423], [1058, 430], [1048, 432], [1037, 426], [1032, 416]], "redness_pct": 0, "occlusion_pct": 25, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 97, "description": "Small green fruit tucked against the left edge of the large red fruit in the right cluster. Sepals obscure its upper surface and the red fruit hides part of its right side."}
{"bbox": [1150.0, 326.0, 1181.0, 364.0], "polygon": [[1157, 332], [1167, 334], [1174, 338], [1178, 348], [1177, 359], [1170, 363], [1162, 359], [1156, 350], [1152, 340]], "redness_pct": 0, "occlusion_pct": 42, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 94, "description": "Tiny green fruit above the far-right large fruit. It is partly hidden behind hanging sepals and a nearly vertical stem, leaving a narrow seeded body visible."}
{"bbox": [160.0, 253.0, 193.0, 291.0], "polygon": [[168, 259], [179, 263], [187, 258], [191, 267], [187, 279], [180, 289], [172, 290], [166, 284], [162, 274]], "redness_pct": 0, "occlusion_pct": 15, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 99, "description": "Tiny dark green fruit hanging by itself on the far left below the upper cluster. Its pointed seeded body is visible under a broad pale green calyx."}
{"bbox": [484.0, 110.0, 515.0, 143.0], "polygon": [[491, 114], [500, 118], [508, 114], [512, 122], [511, 132], [504, 140], [495, 142], [488, 137], [485, 128]], "redness_pct": 0, "occlusion_pct": 23, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 96, "description": "Tiny green developing fruit above and right of the large red-and-cream fruit left of centre. Brown dried flower parts and small sepals surround its upper edge."}
```

</details>

<details markdown="1">
<summary><strong>GPT-6 Astra</strong> on <code>private-2</code> — 21 fruit reported</summary>

```json
{"bbox": [89.0, 337.0, 190.0, 419.0], "polygon": [[94, 352], [108, 344], [121, 355], [140, 365], [158, 349], [178, 354], [188, 369], [185, 390], [172, 407], [150, 417], [127, 414], [106, 403], [94, 385], [90, 368]], "redness_pct": 88, "occlusion_pct": 17, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Large red strawberry on the left, with yellowish patches around its shoulders. Broad sepals cover the upper surface, while the lower body is clearly exposed."}
{"bbox": [179.0, 420.0, 265.0, 522.0], "polygon": [[185, 433], [199, 430], [216, 441], [235, 429], [251, 432], [261, 450], [257, 474], [245, 496], [229, 516], [219, 521], [204, 508], [192, 484], [184, 458]], "redness_pct": 0, "occlusion_pct": 12, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 100, "description": "Large cream-white strawberry in the foreground with green shoulders and prominent dark seeds. Its top is covered by sepals, and it hangs closely beside another pale fruit."}
{"bbox": [254.0, 424.0, 335.0, 513.0], "polygon": [[260, 438], [275, 440], [291, 433], [309, 439], [323, 443], [334, 458], [328, 478], [315, 495], [298, 509], [289, 513], [275, 501], [265, 483], [260, 461]], "redness_pct": 0, "occlusion_pct": 13, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 100, "description": "Pale green to white strawberry immediately right of the large white foreground fruit. Its pointed lower body is exposed, with green sepals overlapping the shoulders."}
{"bbox": [196.0, 473.0, 283.0, 559.0], "polygon": [[198, 506], [210, 516], [220, 522], [233, 513], [247, 495], [257, 492], [263, 502], [276, 500], [280, 518], [269, 539], [251, 551], [231, 558], [215, 549], [206, 530]], "redness_pct": 91, "occlusion_pct": 44, "calyx_visible": true, "peduncle_visible": false, "graspable": false, "confidence_pct": 100, "description": "Deep red strawberry tucked behind the two large pale foreground fruits. Most of its shoulders are hidden, but a broad red lower body projects below them."}
{"bbox": [242.0, 566.0, 313.0, 649.0], "polygon": [[247, 582], [260, 585], [267, 598], [280, 592], [291, 604], [306, 605], [311, 619], [301, 636], [284, 645], [265, 648], [251, 640], [244, 624], [243, 603]], "redness_pct": 3, "occlusion_pct": 23, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Low-hanging pale green strawberry near the bottom center, with a faint pinkish tan blush. Large curled sepals cover much of its upper surface."}
{"bbox": [299.0, 319.0, 373.0, 403.0], "polygon": [[302, 337], [315, 333], [326, 341], [340, 337], [350, 330], [364, 337], [371, 349], [364, 370], [351, 391], [338, 402], [326, 397], [314, 381], [305, 361]], "redness_pct": 97, "occlusion_pct": 16, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Dark red elongated strawberry near the center-right, behind the foreground cluster. Its shoulders are partly covered by its calyx, but the pointed body is well exposed."}
{"bbox": [136.0, 421.0, 187.0, 480.0], "polygon": [[138, 444], [146, 436], [156, 442], [168, 440], [179, 434], [185, 450], [178, 465], [164, 477], [152, 473], [143, 460]], "redness_pct": 0, "occlusion_pct": 49, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 94, "description": "Small green strawberry beneath the large left red fruit and just left of the large white fruit. Broad sepals and the neighboring fruit conceal much of its body."}
{"bbox": [384.0, 439.0, 432.0, 495.0], "polygon": [[389, 451], [399, 445], [409, 449], [420, 449], [429, 456], [431, 470], [423, 483], [410, 494], [400, 491], [391, 479], [385, 463]], "redness_pct": 98, "occlusion_pct": 9, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Small deep red strawberry hanging by itself in the lower-right background. Its rounded shoulders narrow to a downward-pointing tip."}
{"bbox": [354.0, 282.0, 400.0, 337.0], "polygon": [[360, 293], [372, 294], [384, 305], [397, 308], [392, 322], [382, 334], [374, 335], [364, 325], [359, 310]], "redness_pct": 89, "occlusion_pct": 42, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 97, "description": "Red strawberry behind and above the central dark red fruit. An adjacent pale red fruit and crossing stems conceal its upper portion."}
{"bbox": [370.0, 269.0, 407.0, 311.0], "polygon": [[374, 280], [384, 280], [395, 278], [403, 284], [405, 294], [399, 307], [390, 311], [381, 302], [375, 292]], "redness_pct": 54, "occlusion_pct": 24, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Small peach-red strawberry high in the middle-right cluster. It has pale shoulders, prominent seeds, and green sepals across its top."}
{"bbox": [429.0, 400.0, 467.0, 449.0], "polygon": [[436, 404], [447, 402], [458, 409], [466, 421], [462, 434], [451, 445], [440, 449], [432, 440], [430, 425]], "redness_pct": 15, "occlusion_pct": 8, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 100, "description": "Cream to pale pink strawberry in the right background, directly above and right of the lowest red hanging fruit. It has a faint orange blush and an exposed tapered body."}
{"bbox": [472.0, 328.0, 510.0, 374.0], "polygon": [[476, 339], [486, 342], [496, 337], [505, 340], [509, 350], [503, 365], [492, 374], [483, 370], [477, 359], [473, 347]], "redness_pct": 0, "occlusion_pct": 18, "calyx_visible": true, "peduncle_visible": true, "graspable": true, "confidence_pct": 100, "description": "Green strawberry on the right with conspicuous dark seeds. Its body hangs clearly below a spreading calyx."}
{"bbox": [373.0, 359.0, 413.0, 402.0], "polygon": [[379, 370], [389, 373], [400, 366], [409, 370], [412, 382], [405, 396], [391, 401], [378, 398], [374, 387]], "redness_pct": 0, "occlusion_pct": 23, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 100, "description": "Pale green strawberry below and right of the central red fruit. Its upper edge is obscured by sepals and crossing plant growth."}
{"bbox": [409.0, 379.0, 453.0, 427.0], "polygon": [[420, 385], [432, 382], [445, 386], [452, 394], [449, 407], [438, 405], [430, 412], [426, 426], [417, 424], [411, 415], [411, 401]], "redness_pct": 91, "occlusion_pct": 34, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 99, "description": "Red strawberry behind the pale pink fruit on the right. A vertical stem crosses it, and the pale foreground fruit hides its lower-right portion."}
{"bbox": [414.0, 281.0, 448.0, 317.0], "polygon": [[421, 291], [429, 296], [440, 291], [445, 297], [440, 309], [432, 316], [424, 311], [417, 302]], "redness_pct": 0, "occlusion_pct": 43, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 95, "description": "Small green strawberry beneath a prominent star-shaped calyx in the upper-right cluster. The sepals conceal much of the shoulders and leave a narrow green body visible below."}
{"bbox": [335.0, 231.0, 371.0, 265.0], "polygon": [[337, 249], [346, 247], [352, 242], [358, 242], [365, 237], [370, 242], [367, 253], [356, 261], [345, 260]], "redness_pct": 83, "occlusion_pct": 61, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 97, "description": "Partly hidden red strawberry directly beneath the lower edge of the large central leaf. Only an irregular band of red flesh is exposed."}
{"bbox": [374.0, 233.0, 399.0, 261.0], "polygon": [[377, 240], [385, 235], [393, 239], [398, 246], [397, 255], [389, 260], [379, 257], [375, 250]], "redness_pct": 79, "occlusion_pct": 31, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 97, "description": "Small orange-red strawberry to the right of the other fruit under the large leaf. A foreground stem and foliage crowd its upper-left edge."}
{"bbox": [433.0, 325.0, 459.0, 359.0], "polygon": [[438, 337], [445, 339], [451, 335], [457, 341], [454, 352], [446, 359], [439, 353], [435, 345]], "redness_pct": 0, "occlusion_pct": 39, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 92, "description": "Small green strawberry in the background between the pale upper fruit and the red-and-pink lower pair. Sepals and slender stems obscure its shoulders."}
{"bbox": [523.0, 289.0, 547.0, 333.0], "polygon": [[531, 294], [539, 292], [539, 330], [532, 330], [528, 319], [525, 306]], "redness_pct": 98, "occlusion_pct": 40, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 99, "description": "Dark red strawberry cut off by the right image edge near the upper-right cluster. Only the left side of its elongated body is visible."}
{"bbox": [522.0, 379.0, 545.0, 416.0], "polygon": [[529, 385], [537, 386], [539, 391], [539, 412], [533, 414], [526, 410], [523, 401], [525, 391]], "redness_pct": 86, "occlusion_pct": 29, "calyx_visible": true, "peduncle_visible": true, "graspable": false, "confidence_pct": 98, "description": "Small red strawberry clipped by the right edge below the larger green fruit. Its upper surface is partly covered by green sepals."}
{"bbox": [341.0, 530.0, 360.0, 550.0], "polygon": [[345, 534], [351, 530], [357, 534], [360, 540], [357, 547], [351, 550], [345, 547], [342, 541]], "redness_pct": 92, "occlusion_pct": 4, "calyx_visible": false, "peduncle_visible": false, "graspable": false, "confidence_pct": 84, "description": "Tiny blurred red fruit lying on the dark ground covering behind the low foreground stems. It appears detached from the plant."}
```

</details>

Even the runner-up models' output shows the flexibility that makes this paradigm interesting. On `private-2`, GLM distinguishes a "[p]rominent bright, evenly red conical fruit hanging in the centre-right on a long bare green stem … [a]lmost fully exposed and an easy pick" (redness 96%) from the "[v]ery small green fruit with a green calyx hanging low between larger neighbours, partly screened by stems; too small and crowded for a gripper" beside it (redness 2%), flags which fruit a gripper could pick right now, and traces each visible surface around the occluding leaves — all from the same prompt that produced the StrawDI numbers, with no task-specific anything. These are observed outputs, not independently validated horticultural or grasping judgments; the private scenes establish transfer, not accuracy. But the connection between an object, its appearance, and a proposed action is the useful feature: a detection supports counting, a colour description supports user-defined ripeness rules, and an occlusion description can guide which fruit an operator inspects next.

## Current limitations and future directions

### Latency and cost are far from a local detector

Recorded mean call time per scheduled image: **76.2 s (Kimi K3), 89.4 s (GLM), 25.5 s (DeepSeek), 48.4 s (GPT)**, including CLI overhead and recorded retries. For scale, Ultralytics reports **1.5–11.3 ms** for YOLO11 on a T4 GPU with TensorRT — different hardware, resolution, and workload, so not a controlled comparison, but the gap is four orders of magnitude. [12]

GPT averaged **15,336 input and 944 output tokens per scheduled image**. At OpenAI's listed standard rates — $10 / $1 / $50 per million input / cached-input / output tokens, checked September 15, 2026 — that is **≈$0.20 per image** uncached or **≈$0.12** with recorded cache reads ($40.1 vs $23.9 per 200 images); GLM's recorded batch cost was $22.86 per 200 images at its provider's rates. These are token-based estimates, not invoices. [13] For an occasional scene audit this may be acceptable; for continuous video, the first things I would test are request frequency, output verbosity, and distillation into a local model.

![Recorded input and output token usage and mean call time for the four configurations.](/assets/fruit_detection_is_solved_by_vlms/tokens_latency.png)

*Figure 9. Means across all 200 scheduled images per model. Output includes reasoning tokens; GLM's input figure reflects its provider's accounting (cached tokens are not reported as input). Token accounting and CLI overhead differ across providers — these are operational measurements of the configurations used. Control calls excluded.*

### Edge deployable VLMs need their own evaluation

We have not demonstrated VLM detection on an edge device. The natural next experiment is to run locally servable VLMs such as **Qwen3.8-27B** and smaller vision-capable Qwen variants — under the same protocol, measuring memory, power, latency, and localization after quantization on the intended hardware. A model that fits in memory still has to answer within the robot's operating budget. [14]

### Looking Forward to More Capable VLMs
Another important limitation today is that this level of performance is still concentrated in only a small number of frontier models. In our experiments, GPT-6 Astra is clearly ahead of the other VLMs we tested, which means that the current result is still strongly dependent on a single provider and model family.

I expect this to change quickly. As the next generation of VLMs arrives — including models from other commercial providers and open-source communities — it will be important to see whether Astra-level visual detection becomes a broader capability rather than an isolated result. Having multiple models reach this level would improve reproducibility, reduce provider dependence, and make deployment choices more flexible in terms of cost, latency, privacy, and local inference.

## Future questions and conclusion

This experiment answers one question, but it raises several others that I think are more interesting going forward.

1. **How much of this capability can eventually run locally?**
   GPT-6 Astra is powerful, but it is still far from the latency, cost, and deployment characteristics of a local detector. An important next step is to test future edge-capable VLMs on the same images and prompts and see how much of this detection and spatial reasoning capability can survive local deployment.

2. **Which of the richer outputs are actually useful?**
   A VLM can return much more than a class label or mask: descriptions, redness, occlusion, calyx and stem visibility, graspability, confidence, and potentially many other attributes. The next question is which of these outputs are reliable enough to matter in real applications. They should ultimately be validated against expert judgments, downstream decision making, and, for robotic harvesting, actual robot outcomes.

3. **Can a strong VLM teach a much smaller model?**
   If a frontier VLM can generate high-quality detections and polygon masks zero-shot, it may also serve as a data engine for training smaller and faster task-specific models. One possible direction is to use VLM predictions as candidate annotations, review uncertain cases, and train a compact student model that can run locally. Whether such a student can retain the generalization ability of the teacher across new farms and environments remains an open question.

So, is fruit detection solved?

The strongest result in this article is that a general-purpose VLM, given only raw pixels and a written instruction, can produce strawberry detections with both bounding boxes and pixel-level outlines that, on the StrawDI test split, exceed the segmentation mAP of a fully supervised specialist and essentially match its per-instance IoU — all without task-specific training.

That changes how I would approach a new fruit-perception problem. Instead of assuming that every new orchard, crop variety, or environment requires another cycle of data collection, annotation, and model training, I would now first test what a frontier VLM can already do zero-shot.

At the same time, important questions remain: performance on the smallest fruit, latency and cost, robustness across truly independent farms and environments, local deployment, and whether this level of capability will become common across future VLMs rather than remaining concentrated in a few frontier models.

**Fruit detection is not universally solved. But the starting point has changed dramatically: instead of asking how to train a detector from scratch, we can increasingly ask how far a general-purpose visual model can already take us — and what still needs to be built on top of it.**


---


**Dataset acknowledgement:** Kindly provided by the StrawDI Team (see [the official dataset website][strawdi]).

## References

[1] Fei, Z., Olenskyj, A. G., Bailey, B. N., & Earles, M. (2021). Enlisting 3D crop models and GANs for more data-efficient and generalizable fruit detection. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 1269–1277.

[2] Wang, Y., Fei, Z., Li, R., & Ying, Y. (2025). Learn from foundation model: Fruit detection model without manual annotation. *Pattern Recognition*, 112799.

[3] Wang, Y., Li, W., Ying, Y., & Fei, Z. (2026). GEAR-Seg: A Grounded Explainable Agent for Reasoning Segmentation and Data Engine. *arXiv preprint* arXiv:2607.00544.

[4] Ranasinghe, K., et al. (2024). Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[5] Schaumloffel, L., et al. (2026). Mechanisms of Object Localization in Vision-Language Models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[6] SOUBench: Benchmarking Small-Object Understanding in Multimodal Large Language Models. (2026).

[7] OpenAI. (2026). GPT-6 Astra: The Next Generation in Intelligence for Work. https://openai.com/index/gpt-6-astra-next-generation-work/

[8] Su et al. (2026). GPT-6 Astra as an Embodied Policy. https://anonymous-report-421.github.io/public-website/?lang=en&view=1

[9] Robocurve. (2026). GPT-6 Astra on Robotic Manipulation. https://openai.robocurve.org/gpt-6-astra/

[10] Pérez-Borrero, I., Marín-Santos, D., Gegúndez-Arias, M. E., & Cortés-Ancos, E. (2020). A fast and accurate deep learning method for strawberry instance segmentation. *Computers and Electronics in Agriculture*, 178, 105736. https://doi.org/10.1016/j.compag.2020.105736

[11] COCO Consortium. Common Objects in Context (COCO) evaluation API (`cocoeval.py`). https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

[12] Ultralytics. YOLO11 performance benchmarks. https://docs.ultralytics.com/models/yolo11/

[13] OpenAI. GPT-6 Astra API pricing. https://developers.openai.com/api/docs/models/gpt-6-astra

[14] Qwen Team. Qwen3.8 documentation. https://github.com/QwenLM/Qwen3.8/blob/main/README.md

[strawdi]: https://strawdi.github.io/
[paper]: https://doi.org/10.1016/j.compag.2020.105736
