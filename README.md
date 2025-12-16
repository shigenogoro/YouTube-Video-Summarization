# BART- and BERT-Based Chaptering Pipeline: Evaluating Fixed-Window and Semantic Segmentation for Temporal Alignment on Meeting and YouTube Transcripts

**Authors:** Sheng-Kai Wen, Mohammad Derakshi  
**Date:** December 2025

## Overview

This project implements a modular pipeline to generate concise, timestamp-aligned video chapters from ASR transcripts. It addresses the challenge of navigating long-form video content (lectures, tutorials, meetings) by converting raw transcripts into short, clickable chapter titles linked to specific time ranges.

The system was evaluated on two datasets:
1.  **MeetingBank** (Phase 1): Focused on meeting transcripts.
2.  **VidChapter-7M** (Phase 2): A subset of 972 YouTube videos, comparing duration-based and semantic segmentation strategies.

## Pipeline Architecture

![System Pipeline](report/pipeline.png)
*Figure 1: The hierarchical chaptering pipeline. (1) The input ASR transcript is segmented using either a Fixed-Window or Semantic strategy. (2) A fine-tuned BART model generates a concise chapter title for each segment. (3) A BERT-based alignment classifier verifies and aligns the generated titles to the video timeline to ensure temporal accuracy.*

## Key Methodologies

### 1. Segmentation
We explored three strategies to divide long transcripts into manageable chunks:
*   **Fixed-Window (K=6):** Splits the transcript into 6 equal-duration segments. This proved to be a robust baseline.
*   **Semantic Segmentation (Adaptive K):** Predicts the number of segments ($K$) using a regressor and splits based on Sentence-BERT embedding similarity.
*   **Semantic Segmentation (Threshold):** Splits based on a fixed cosine similarity threshold.

### 2. Summarization (Chapter Generation)
*   **Model:** BART (base) fine-tuned on segment-title pairs.
*   **Task:** Generate a short, descriptive chapter title for each transcript segment.

### 3. Temporal Alignment
*   **Model:** BERT-based binary classifier.
*   **Task:** Predicts whether a generated summary sentence matches a specific transcript segment, enabling precise timestamp alignment.

## Results

### Phase 1: MeetingBank (Meeting Summarization)
In Phase 1, we validated the divide-and-conquer approach on structured meeting transcripts. Fine-tuning BART yielded significant improvements over the pretrained baseline.

**Summarization Performance**

| Metric | Fine-tuned BART | Pretrained BART | Gain |
| :--- | :--- | :--- | :--- |
| **ROUGE-1 (Segment)** | **64.07%** | 35.89% | +28.18 |
| **ROUGE-L (Segment)** | **61.51%** | 31.01% | +30.49 |
| **BERTScore F1** | **91.44%** | - | - |

**Alignment Performance**

| Metric | Accuracy |
| :--- | :--- |
| **Alignment Accuracy** | **80.43%** |

### Phase 2: VidChapter-7M (YouTube Chaptering)
In Phase 2, we applied the pipeline to open-domain YouTube videos. Experiments showed that the **Fixed-Window** segmentation strategy surprisingly outperformed adaptive semantic approaches in both summarization quality and alignment accuracy.

**Summarization & Alignment**

| Metric | Fixed-Window (K=6) | Semantic (Adaptive) | Semantic (Threshold) |
| :--- | :--- | :--- | :--- |
| **ROUGE-1** | **14.80%** | 12.81% | 3.57% |
| **BERTScore F1** | **86.99%** | 86.56% | 84.88% |
| **Temporal F1 (±30s)** | **65.35%** | 62.74% | 39.51% |

**Segment Number Prediction (Adaptive K)**

To support the adaptive semantic strategy, we trained a regressor to predict the optimal number of chapters ($K$) for each video.

| Metric | Value |
| :--- | :--- |
| **MAE** | 4.58 chapters |
| **Exact Match Accuracy** | 10.29% |
| **Weighted Recall** | 10.29% |

## Repository Structure

*   `notebooks/`: Jupyter notebooks for Phase 1 (MeetingBank) and Phase 2 (VidChapter-7M) experiments.
*   `models/`: Saved checkpoints for the fine-tuned BART summarizer and BERT alignment model.
*   `data/`: Evaluation results and metrics.
*   `report/`: Detailed project reports and figures.
