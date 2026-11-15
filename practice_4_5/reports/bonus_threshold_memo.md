# Task 5B: Operating Threshold & Pipeline Cost Memo

### 1. Pipeline error cost asymmetry
* False Accept (bad photo classified as good, Cost = 10x): A blurred, glared, or occluded document slips past Stage 1, consumes 1.5-3.5 seconds of downstream PaddleOCR compute, produces corrupted text fields (incorrect passport serial numbers or INN digits), and forces a human back-office operator to manually review, flag, and reject the submission.
* False Reject (good photo classified as bad, Cost = 1x): The Telegram bot immediately prompts the user within 18.86 ms to retake the photo in better lighting-a minor user inconvenience with zero OCR corruption and zero operator rework.
* Because False Accepts impose downstream OCR compute waste and human operator rework, we penalize False Accepts 10:1 over False Rejects and shift the P(bad) operating threshold below 0.50.


### 2. Validation threshold sweep table

| Threshold | Precision (bad) | Recall (bad) | F1 (bad) | False Accepts | False Rejects | Weighted Cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.7988 | 1.0000 | 0.8882 | 0 | 34 | 34.0 |
| 0.15 | 0.8491 | 1.0000 | 0.9184 | 0 | 24 | 24.0 |
| 0.20 | 0.9306 | 0.9926 | 0.9606 | 1 | 10 | 20.0 |
| 0.25 (Chosen) | 1.0000 | 0.9926 | 0.9963 | 1 | 0 | 10.0 |
| 0.30 | 1.0000 | 0.9926 | 0.9963 | 1 | 0 | 10.0 |
| 0.35 | 1.0000 | 0.9852 | 0.9925 | 2 | 0 | 20.0 |
| 0.40 | 1.0000 | 0.9852 | 0.9925 | 2 | 0 | 20.0 |
| 0.45 | 1.0000 | 0.9778 | 0.9888 | 3 | 0 | 30.0 |
| 0.50 (Default) | 1.0000 | 0.9778 | 0.9888 | 3 | 0 | 30.0 |
| 0.55 | 1.0000 | 0.9630 | 0.9811 | 5 | 0 | 50.0 |
| 0.60 | 1.0000 | 0.9630 | 0.9811 | 5 | 0 | 50.0 |
| 0.65 | 1.0000 | 0.9481 | 0.9734 | 7 | 0 | 70.0 |
| 0.70 | 1.0000 | 0.9481 | 0.9734 | 7 | 0 | 70.0 |
| 0.75 | 1.0000 | 0.9481 | 0.9734 | 7 | 0 | 70.0 |
| 0.80 | 1.0000 | 0.9407 | 0.9695 | 8 | 0 | 80.0 |
| 0.85 | 1.0000 | 0.9333 | 0.9655 | 9 | 0 | 90.0 |
| 0.90 | 1.0000 | 0.9111 | 0.9535 | 12 | 0 | 120.0 |


### 3. Explanation checkpoint
* Selected operating point (τ = 0.25) sits at the Pareto-optimal elbow of the Precision–Recall curve (F1(bad) = 0.9963). Pushing lower (τ = 0.20) begins triggering False Rejects (10 clean documents rejected, dropping precision to 0.9306), whereas staying at τ = 0.50 allows 3 degraded images through.
* Quantified operator impact: Moving from τ = 0.50 to τ = 0.25 reduces False Accepts on validation from 3 images (2.22%) down to 1 image (0.74%) - eliminating 66.7% of manual operator rework and wasted OCR cycles while keeping user retake friction at 0 False Rejects (0.00%). It also creates a wider safety buffer against the 2.8-5.8% synthetic-to-real confidence shift in production.
* Why test was not used: Tuning τ on the held-out test split leaks test-set error structure into model selection, optimistically biasing final test metrics and invalidating unbiased generalization estimates.