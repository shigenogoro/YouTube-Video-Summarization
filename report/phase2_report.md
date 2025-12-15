# Phase 2 Report — VidChapter-7M Experiments

## Abstract

This Phase 2 report summarizes the VidChapter-7M experiments (video chapter generation and alignment) implemented in `notebooks/phase2_VidChapter-7M/02_VidChapter_Pipeline.ipynb` and `03_VidChapter_Segment.ipynb`. We explored three segmentation strategies: Fixed-Window (Constant K), Semantic Segmentation (Adaptive K), and Semantic Segmentation (Threshold). The results show that the Fixed-Window approach outperforms both semantic segmentation approaches in summarization (ROUGE) and alignment (Temporal F1) metrics on this dataset.

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
  3. **Semantic Segmentation (Threshold)**:
     - Splits the transcript whenever the cosine similarity between consecutive Sentence-BERT embeddings falls below a fixed threshold (0.4).

- **Temporal Alignment**
  - Approach: For each generated chapter title, find the best matching transcript segment using a BERT-based classifier trained on positive (ground truth) and negative (random) pairs.
  - Metric: Temporal F1 (±15s and ±30s tolerance).

## Evaluation

### Summarization Performance (ROUGE & BERTScore)

| Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|----------|---------|---------|---------|--------------|
| Fixed-Window (K=6) | 14.80% | 6.56% | 14.31% | 85.38% |
| Semantic-Seg (Adaptive K) | 12.81% | 5.10% | 12.51% | 85.16% |
| Semantic-Seg (Threshold=0.4) | 3.57% | 0.91% | 3.50% | 83.16% |

### Alignment Performance (Temporal F1)

| Strategy | F1 (±15s) | F1 (±30s) |
|----------|-----------|-----------|
| Fixed-Window (K=6) | 65.35% | 65.52% |
| Semantic-Seg (Adaptive K) | 62.74% | 64.66% |
| Semantic-Seg (Threshold=0.4) | 39.51% | 41.78% |

### Segment Number Prediction (Adaptive K)

- **Model**: GradientBoostingRegressor
- **MAE**: 4.58 chapters
- **Accuracy (Exact Match)**: 10.29%
- **Recall (Weighted)**: 10.29%

**Result & Discussion**

- **Summarization**: The Fixed-Window approach achieved the highest ROUGE scores. The Semantic Segmentation (Adaptive K) followed, while the Threshold-based method performed significantly worse. This suggests that forcing a specific number of segments (either fixed or predicted) is more effective than relying solely on a similarity threshold, which might produce too many or too few segments depending on the video's semantic density.
- **Alignment**: The Fixed-Window method also led in temporal alignment. The Semantic Threshold method's poor performance in summarization likely cascaded into the alignment task, as the generated titles were of lower quality or the segments were too fragmented to align correctly.
- **Adaptive K Prediction**: The segment number predictor (GradientBoostingRegressor) was used to dynamically determine $K$ for the semantic approach. While it provides a tailored $K$ for each video, the overall pipeline performance was still lower than the fixed $K=6$ baseline, indicating that the predicted $K$ might not always be optimal for the summarization model or that the semantic split points are not aligning well with the ground truth chapters.

**Future Work**

- **Improved Thresholding**: The Semantic Threshold method could be improved by dynamically selecting the threshold per video or using a more sophisticated boundary detection algorithm (e.g., C99, TextTiling).
- **Hybrid Approaches**: Combining fixed-window constraints with semantic boundaries (e.g., allowing semantic splits only within a certain time window) could balance the robustness of fixed windows with the content-awareness of semantic segmentation.

**Conclusion**

Phase 2 demonstrated the feasibility of generating video chapters and aligning them to the timeline. The Fixed-Window baseline proved to be the most robust strategy, outperforming both adaptive and threshold-based semantic segmentation methods. The results highlight the challenge of unsupervised segmentation for chapter generation and suggest that simple heuristics can be surprisingly effective baselines.

**Files referenced**

- Pipeline Notebook: `notebooks/phase2_VidChapter-7M/02_VidChapter_Pipeline.ipynb`
- Segmentation Notebook: `notebooks/phase2_VidChapter-7M/03_VidChapter_Segment.ipynb`
- Models: `models/vidchapter_bart_best`, `models/vidchapter_alignment`
