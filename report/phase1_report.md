# Phase 1 Report — MeetingBank Experiments

## Abstract

This Phase 1 report summarizes the MeetingBank experiments (divide-and-conquer summarization and temporal alignment) implemented in `notebooks/phase1_meetingBank/01_MeetingBank.ipynb`. Fine-tuning BART yields large segment-level ROUGE gains over the pretrained baseline; meeting-level concatenated summaries remain strong. Temporal-alignment was implemented and alignment results are now available (see Evaluation).

## Experiment

- **Summarization**
  - Approach: divide transcripts into segments, fine-tune a seq2seq model per-segment, decode segment summaries, then concatenate segment summaries into full-meeting summaries for meeting-level evaluation.
  - Model: BART (base) fine-tuned via Hugging Face `Seq2SeqTrainer`. A fine-tuned checkpoint is available at `models/meetingbank/checkpoint-1293/`.

- **Temporal Matching (Temporal Alignment)**
  - Approach: form (summary sentence, transcript segment) pairs, train a binary classifier (BERT) to predict matches, then assign each sentence the highest-scoring segment at inference time.

## Evaluation

- Evaluation results
  - `data/meetingbank/eval/fine_tuned_bart_rouge_scores_segment.csv`
  - `data/meetingbank/eval/pre_trained_bart_rouge_scores_segment.csv`
  - `data/meetingbank/eval/fine_tuned_bart_rouge_scores_meeting.csv`
  - `data/meetingbank/eval/alignment_results.csv`

### Segment-level ROUGE

| Metric     | Fine-tuned (%) | Pretrained (%) | Gain (pts) |
|------------|----------------|----------------|------------|
| ROUGE-1    | 64.07          | 35.89          | 28.18      |
| ROUGE-2    | 54.18          | 24.47          | 29.71      |
| ROUGE-L    | 61.51          | 31.01          | 30.49      |
| ROUGE-Lsum | 61.47          | 30.99          | 30.48      |

Values rounded to 2 decimal places; gains = (fine-tuned − pretrained).

### Meeting-level Performance (concatenated segments)

**ROUGE Scores**

| Metric     | Fine-tuned (%) |
|------------|----------------|
| ROUGE-1    | 62.97          |
| ROUGE-2    | 52.31          |
| ROUGE-L    | 59.83          |
| ROUGE-Lsum | 59.79          |

**BERTScore**

| Metric | Score (%) |
|--------|-----------|
| Precision | 93.37 |
| Recall | 89.67 |
| F1 | 91.44 |

### Temporal Alignment

| Metric | Accuracy |
|--------|----------|
| Alignment Accuracy | 80.43% |

- File: `data/meetingbank/alignment_results.csv` (inspected)
- Alignment accuracy: **80.43%** (0.8043) — fraction of sentence-level alignments where predicted segment matched the ground-truth segment.

**Result & Discussion**

- Summarization: fine-tuning BART on MeetingBank produces very large segment-level ROUGE improvements vs. the pretrained baseline; concatenating segment summaries produces meeting-level ROUGE that remains high, indicating the divide-and-conquer strategy is effective for this dataset.

- Temporal alignment: the implemented BERT-based matching pipeline achieves **~80.43%** alignment accuracy on the evaluated set (value from `data/meetingbank/alignment_results.csv`). This is a strong starting point for sentence→segment matching; further improvements could come from:
  - stronger negative sampling or contrastive training,
  - sequence labeling approaches that consider sentence order context, or
  - joint models that incorporate segment and meeting context.

**Caveats**

- Notebook contains Colab-specific cells (Drive mount, `%cd`) that can change where outputs are written — confirm runs were done in the repo root before relying on file locations.
- Reproducing training requires appropriate compute (GPU recommended) and the notebook's training/eval settings.

**Conclusion**

The Phase 1 MeetingBank experiments produced a clear, large improvement from fine-tuning BART for segment summarization (segment ROUGE gains > 28 ROUGE-1) and maintained strong meeting-level ROUGE after concatenation. The temporal alignment pipeline is implemented and reports **~80.43%** alignment accuracy. Recommended next steps are: tune generation settings (beams, length penalties), conduct human evaluation on full-meeting summaries, and iterate on the alignment model (contrastive loss, context-aware models) to further raise match quality.

**Files referenced**

- Notebook: `notebooks/phase1_meetingBank/01_MeetingBank.ipynb`
- Models: `models/meetingbank/checkpoint-1293/`
- Outputs: files under `data/meetingbank/eval` (predictions, ROUGE CSVs, `alignment_results.csv`).

