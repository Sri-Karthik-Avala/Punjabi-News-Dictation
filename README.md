# Punjabi News Dictation

| | |
| --- | --- |
| Final rank | 1st |
| Domain | Sequence To Sequence |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Creator | aikyatan |
| Solutions submitted | 3 |
| Last submission | 2026-09-16 |

## Problem statement

### Background

Automatic Speech Recognition (ASR) for regional and morphologically rich languages presents unique challenges, particularly in domain-specific broadcast environments like news dictation. Spoken news speech contains diverse named entities, inflections, complex sentence structures, and variable pacing. Delineating phonetic acoustic nuances and transcribing them directly into native orthography without intermediate phonetic representations requires robust sequence-to-sequence acoustic and linguistic processing.

### Overview

Given variable-length audio recordings of spoken Punjabi news dictations, transcribe each utterance into ordered, native Gurmukhi text. The solution must recover the full spoken lexical sequence, including named entities, numbers, and inflected word forms, directly from the audio signal.

This is a **GPU-only sequence-to-sequence challenge**. Learned training, adaptation, and inference must execute on CUDA within a single self-contained offline session. The operational hardware envelope provides one NVIDIA A10G and a strict 90-minute total solution runtime.

### Dataset Information (Public Files)

All assets necessary for training and evaluation are supplied in the public directory. Audio files are provided as mono 16 kHz PCM-16 FLAC files.

```
+-----------------------+--------------------------------------------------------------+
| File / Directory      | Purpose                                                      |
+-----------------------+--------------------------------------------------------------+
| audio/                | Directory of mono 16 kHz PCM-16 FLAC audio recordings.       |
| train.csv             | 715 labeled training examples with reference transcriptions. |
| test.csv              | 185 evaluation queries requiring transcriptions.             |
| sample_submission.csv | Format-example rows demonstrating the required output schema.|
+-----------------------+--------------------------------------------------------------+
```

### Split Protocol and Limits

Preparation deduplicates normalized full transcripts, links records sharing the first four transcript words or an exact transcript, and assigns each entire connected group by a deterministic hash with nominal 20% evaluation probability. The retained split has 715 training rows and 185 evaluation rows. Exact transcripts and shared four-word prefixes cannot cross this boundary. This controls direct phrase copying; it does not establish speaker, recording-session, topic, or paraphrase disjointness. Broadcast vocabulary can recur across splits. Validation must be constructed from training rows and keep equal four-word transcript prefixes together. Evaluation measures transcription within this broadcast domain, not transfer to unseen speakers or domains.

### Audio Rendering

The waveform is converted to mono 16 kHz. Deterministic rendering applies gain 0.90–1.10, Gaussian noise with sampled standard deviation 0.002–0.010, zero to two 10–29-sample transients with offsets in [−0.15,0.15], a 50 or 60 Hz hum of amplitude 0.005, and a 1000–3000 Hz tone of amplitude 0.002. Finally it is multiplied by 0.92, clipped to [−1,1], and encoded as PCM-16 FLAC. These amplitude values are in normalized waveform units. Rendering changes the input audio, not the reference text.

### Feature Schema

`train.csv` **and** `test.csv`

```
+------------------+---------+---------------------------------------------------------+
| Column           | Type    | Description                                             |
+------------------+---------+---------------------------------------------------------+
| id               | String  | Opaque 24-character hexadecimal row identifier.         |
| audio            | String  | Relative path to the FLAC audio file (audio/<id>.flac). |
| duration_seconds | Float   | Decoded audio duration in seconds (at most 20.0s).      |
| prediction       | String  | (Train only) Reference Gurmukhi text transcription.     |
+------------------+---------+---------------------------------------------------------+
```

### Target Text Schema

Predictions must be submitted as plain text in the Gurmukhi script.

- **Script & Formatting:** Transcriptions must use standard Gurmukhi orthography conforming to the conventions shown in the training split. Do not output phonetic Latin transliterations, romanized script, or translations into English.
- **Normalization:** Text will undergo standard Unicode NFC normalization and consecutive whitespace collapsing prior to evaluation.
- **Target Length Limits:** Ground-truth training targets contain between 5 and 45 words and at most 600 characters. Any test prediction exceeding **100 words** or **1,200 characters** will automatically receive a score of **0.0** for that row.

An illustrative target from the training set:

```
ਕੁਲ ਦੁਨੀਆਂ ਚ ਤਾਂ ਕਰੋੜ ਤੋਂ ਵੱਧ ਹੋ ਗਏ ਹੋਣਗੇ
```

### Evaluation Metrics

Word edit similarity penalizes missing, extra and substituted lexical units. Character edit similarity gives graded credit for partially correct Gurmukhi spellings and inflections within an otherwise incorrect word. Equal weighting balances whole-word recovery and orthographic fidelity. This is a text-transcription metric, not a direct acoustic or phonetic-distance measurement: Unicode codepoints are compared without phoneme alignment, and whitespace and NFC normalization are the only invariances.

Submissions are evaluated row-by-row by comparing the normalized predicted text against the reference transcription. The evaluation balances word-level structural consistency and character-level phonetic alignment.

Both candidate and reference strings are preprocessed by collapsing whitespace and applying Unicode NFC normalization.

1. Edit Similarity ($E$)

For two token sequences $a$ and $b$ (whether lists of words or sequences of characters), edit similarity is computed using unit-cost Levenshtein distance:

$E(a, b) = \max\left(0, 1 - \frac{\text{Levenshtein}(a, b)}{\max(\text{len}(a), \text{len}(b))}\right)$

- If both $a$ and $b$ are empty, $E(a, b) = 1.0$.
- If exactly one sequence is empty, $E(a, b) = 0.0$.

### 2. Word and Character Metrics

- **Word Edit Similarity ($S_{\text{word}}$):** Evaluated by whitespace-tokenizing strings into word arrays: $S_{\text{word}} = E(P_{\text{words}}, T_{\text{words}})$
- **Character Edit Similarity ($S_{\text{char}}$):** Evaluated directly on Unicode codepoints: $S_{\text{char}} = E(P, T)$

### 3. Row Score and Final Score

The overall row score is the unweighted average of word and character similarities:

$\text{Row Score} = 0.5 \times S_{\text{word}} + 0.5 \times S_{\text{char}}$

The final competition score is the arithmetic mean of all row scores across the test set:

$\text{Final Score} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \text{Row Score}_i$

Scores range from $0.0$ to $1.0$ (higher is better).

### Sample Submission Format

Submit a UTF-8 encoded CSV file containing exactly two columns in this order: `id,prediction`. Include one row per test ID with no missing, extra, or duplicate IDs. Quote strings using standard CSV conventions to ensure punctuation and commas do not corrupt row formatting.

```
id,prediction
4cbde45743f8bd7e1b57f67e,"ਕੁਲ ਦੁਨੀਆਂ ਚ ਤਾਂ ਕਰੋੜ ਤੋਂ ਵੱਧ ਹੋ ਗਏ ਹੋਣਗੇ"
```

> **Parsing Bounds & Rejection:**
>
>
>
> - Missing, extra, duplicate, or unaligned IDs, incorrect column headers, or files exceeding 256 MB raise an informative `ValueError` and reject the submission.
> - Predictions longer than 100 words or 1,200 characters score **0.0** for that row.
> - Individual prediction cells exceeding 65,536 characters trigger an invalid cell error and score **0.0** for the row.
> - Row ordering does not affect grading.

### What Not To Use

To ensure rigorous evaluation of learning efficiency and offline speech recognition:

- **Offline Execution Only:** No internet access, hosted inference APIs, external cloud lookups, or reverse-engineering from external archives.
- **No External Datasets:** Solutions must rely strictly on the supplied training data for task supervision. External speech datasets or outside text corpora cannot be downloaded during the run.
- **Pretrained Checkpoints:** Generic pretrained weights (such as foundational ASR checkpoints) may only be used if explicitly pre-provisioned in the isolated environment's allowlist. Domain-specific, fine-tuned, or private checkpoints are strictly prohibited.
- **No Exploitation:** Manual labeling of test samples, hard-coded lookup dictionaries, or test ID conditioning is forbidden.
- **Hardware Envelope:** All training/adaptation and neural inference must finish within 90 minutes on the designated GPU environment (1 NVIDIA A10G)
