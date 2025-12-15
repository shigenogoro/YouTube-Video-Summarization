# Phase 2 Report — VidChapter-7M Experiments

## Abstract

This Phase 2 report summarizes the VidChapter-7M experiments (video chapter generation and alignment) implemented in `notebooks/phase2_VidChapter-7M/02_VidChapter_Pipeline.ipynb` and `03_VidChapter_Segment_version11.ipynb`. We explored two segmentation strategies: Fixed-Window (Constant K) and Semantic Segmentation (Adaptive K). The results show that the Fixed-Window approach slightly outperforms the Semantic Segmentation approach in both summarization (ROUGE) and alignment (Temporal F1) metrics on this dataset.

## Experiment

- **Dataset**
  - VidChapter-7M subset (972 videos with both Chapters and ASR data).
  - Split: 90% Training (~875 videos), 10% Testing (~97 videos).

- **Summarization**
  - Model: BART (base) fine-tuned on the training split.
  - Task: Generate a chapter title given a transcript segment.

- **Segmentation Strategies**
  1. **Fixed-Window (Constant K=6)**: Splits the transcript into 6 equal-duration segments.
  2. **Semantic Segmentation (Adaptive K)**:
     - **K-Prediction**: A `GradientBoostingRegressor` predicts the number of segments ($K$) based on video duration, word count, number of ASR lines (subtitle blocks), and words per second.
     - **Segmentation**: Uses Sentence-BERT embeddings to find $K-1$ split points with the lowest semantic similarity.

- **Temporal Alignment**
  - Approach: For each generated chapter title, find the best matching transcript segment using a BERT-based classifier trained on positive (ground truth) and negative (random) pairs.
  - Metric: Temporal F1 (±15s and ±30s tolerance).

## Evaluation

### Summarization Performance (ROUGE & BERTScore)

| Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|----------|---------|---------|---------|--------------|
| Fixed-Window (K=6) | 15.46% | 7.15% | 15.19% | 85.16% |
| Semantic-Seg (Adaptive K) | 14.16% | 4.81% | 13.76% | 84.94% |

### Alignment Performance (Temporal F1)

| Strategy | F1 (±15s) | F1 (±30s) |
|----------|-----------|-----------|
| Fixed-Window (K=6) | 68.75% | 69.10% |
| Semantic-Seg (Adaptive K) | 66.54% | 68.01% |

### Segment Number Prediction (Adaptive K)

- **Model**: GradientBoostingRegressor
- **MAE**: 4.58 chapters
- **Accuracy (Exact Match)**: 10.29%
- **Recall (Weighted)**: 10.29%

**Result & Discussion**

- **Summarization**: The Fixed-Window approach achieved higher ROUGE scores compared to the Semantic Segmentation approach. This suggests that for this specific subset of VidChapter-7M, simple duration-based splitting might be more robust or that the semantic boundaries did not align perfectly with the ground truth chapter boundaries used for training.
- **Alignment**: The Fixed-Window method also performed better in temporal alignment. This could be because the fixed windows provide a more consistent coverage of the video, whereas semantic segmentation might create irregular segments that are harder to align with the generated titles.
- **Adaptive K Prediction**: The segment number predictor (GradientBoostingRegressor) was used to dynamically determine $K$ for the semantic approach, but the overall performance was still lower than the fixed $K=6$ baseline.

**Future Work**

- **Semantic Threshold Method**: We planned to explore a "Semantic Threshold" method where segments are created based on a similarity threshold rather than a fixed number $K$. However, due to computational limitations (calculating pairwise similarities and finding optimal thresholds for all videos is expensive), this method was not fully evaluated in this phase. Future work should investigate this approach as it allows for a variable number of segments that naturally fit the content structure.

**Conclusion**

Phase 2 demonstrated the feasibility of generating video chapters and aligning them to the timeline. While the Semantic Segmentation approach is theoretically more appealing, the Fixed-Window baseline proved to be a strong competitor, outperforming the adaptive method in this experimental setup. Future improvements could focus on better segmentation algorithms (e.g., Semantic Threshold) and larger training sets.

**Files referenced**

- Pipeline Notebook: `notebooks/phase2_VidChapter-7M/02_VidChapter_Pipeline.ipynb`
- Segmentation Notebook: `notebooks/phase2_VidChapter-7M/03_VidChapter_Segment_version11.ipynb`
- Models: `models/vidchapter_bart_best`, `models/vidchapter_alignment`
