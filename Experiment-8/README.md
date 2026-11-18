# Experiment 8: Abstractive Text Summarization using mT5 and mBART

**Name:** Rudransh Sogani  
**SAP ID:** 500120440  

## Objective
To implement and evaluate abstractive text summarization using two multilingual Transformer-based sequence-to-sequence models, mT5 (Multilingual Text-to-Text Transfer Transformer) and mBART (Multilingual Bidirectional and Auto-Regressive Transformers), on the Hindi subset of the XL-Sum (Cross-Lingual Summarization) dataset, and to compare their performance using ROUGE-1, ROUGE-2, ROUGE-L, and mBERT scores.

## Procedure & Implementation
This experiment involves multiple phases covering data loading, summary generation with two models, and automatic evaluation:

1.  **Dataset (`8.1`):**
    *   Downloading the Hindi test split of XL-Sum (BBC Hindi news articles with one-line reference summaries) from the Hugging Face Hub <sup>(`hf_hub_download()`, `pd.read_parquet()` / `tarfile` + `.jsonl`)</sup>.
    *   Taking a random chunk of 200 unseen test articles instead of the full test set <sup>(`DataFrame.sample(200)`)</sup>.
    *   Inspecting article and reference summary lengths <sup>(`str.split().str.len()`, `describe()`)</sup>.
    *   Normalizing whitespace and newlines in each article before tokenization <sup>(`WHITESPACE_HANDLER`)</sup>.
    *   Writing a batched summary generation function with input truncation to 512 tokens <sup>(`tokenizer(truncation=True, max_length=512)`, `model.generate()`)</sup>.

2.  **mT5 Summarization (`8.2`):**
    *   Loading mT5-base fine-tuned on all 45 XL-Sum languages <sup>(`AutoModelForSeq2SeqLM.from_pretrained('csebuetnlp/mT5_multilingual_XLSum')`)</sup>.
    *   Generating summaries with beam search <sup>(`num_beams=4`, `max_length=84`, `no_repeat_ngram_size=2`)</sup>.
    *   Freeing GPU memory before loading the next model <sup>(`gc.collect()`, `torch.cuda.empty_cache()`)</sup>.

3.  **mBART Summarization (`8.3`):**
    *   Loading mBART-large-50 fine-tuned on XL-Sum, including Hindi <sup>(`AutoModelForSeq2SeqLM.from_pretrained('mskayacioglu/mbart50-xlsum')`)</sup>.
    *   Setting the Hindi language code for the input and forcing Hindi output <sup>(`src_lang='hi_IN'`, `forced_bos_token_id`)</sup>.
    *   Generating summaries with beam search and repetition control <sup>(`num_beams=5`, `length_penalty=1.1`, `no_repeat_ngram_size=3`)</sup>.

4.  **Evaluation (`8.2`–`8.3`):**
    *   Computing ROUGE-1, ROUGE-2, and ROUGE-L F1 for every article <sup>(`rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])`)</sup>.
    *   Using a custom Devanagari tokenizer, since the default ROUGE tokenizer keeps only English letters and scores all Hindi text as 0 <sup>(`HindiTokenizer`, `re.findall(r'[ऀ-ॿ\w]+')`)</sup>.
    *   Computing semantic similarity between generated and reference summaries with multilingual BERT <sup>(`bert_score.score(model_type='bert-base-multilingual-cased')`)</sup>.

5.  **Comparative Analysis (`8.4`–`8.5`):**
    *   Printing sample articles with the reference, mT5, and mBART summaries side by side.
    *   Tabulating mean ROUGE, mBERT Precision/Recall/F1, average summary length, and generation time <sup>(`pd.DataFrame`)</sup>.
    *   Plotting a grouped bar chart of the metrics and per-article score distributions <sup>(`DataFrame.plot(kind='bar')`, `sns.boxplot()`)</sup>.
    *   Comparing summary lengths and the share of articles where each model wins on ROUGE-L <sup>(`plt.hist()`, `plt.bar()`)</sup>.

## Observations
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | mBERT-P | mBERT-R | mBERT-F1 | Avg Summary Length (words) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mT5** | _ | _ | _ | _ | _ | _ | _ |
| **mBART** | _ | _ | _ | _ | _ | _ | _ |

<sub>Fill in from the Markdown table printed by the last cell of the notebook.</sub>

| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **Dataset Chunk** | 200 Hindi test articles with single-sentence reference summaries | Length statistics |
| **mT5** | mT5-base (~580M params), multilingual T5 text-to-text model | Beam search generation, sample outputs |
| **mBART** | mBART-large-50 (~610M params) with forced `hi_IN` language token | Beam search generation, sample outputs |
| **ROUGE** | Word overlap (unigrams, bigrams, longest common subsequence) with the reference | `rouge_score` with a Hindi tokenizer |
| **mBERT Score** | Meaning-level similarity even when different words are used | BERTScore with `bert-base-multilingual-cased` |
| **Comparison** | Mean scores, per-article score spread, summary length, ROUGE-L win rate | Bar chart, box plot, histograms |

## Result Interpretation & Learnings
*   **Abstractive Summaries:** Both models write new sentences rather than copying sentences from the article, producing short, headline-like Hindi summaries similar to the XL-Sum references.
*   **Pretraining Objectives:** mT5 is pretrained by filling in masked spans of text, while mBART is pretrained as a denoising auto-encoder that rebuilds shuffled and masked sentences. Both suit sequence-to-sequence tasks such as summarization.
*   **Language Control:** mBART needs an explicit language token (`hi_IN`) to generate Hindi, while mT5 treats every task as text-to-text and infers the language from the input.
*   **ROUGE vs. mBERT Score:** ROUGE only rewards exact word overlap, so a correct summary that uses synonyms scores low. mBERT score compares contextual embeddings and captures meaning, giving a fairer view of abstractive summaries.
*   **Hindi-Specific Evaluation:** The standard ROUGE tokenizer removes Devanagari characters and gives a score of 0, so language-aware tokenization is essential when evaluating non-English summaries.
*   **Short References:** Hindi XL-Sum references are a single sentence, which keeps ROUGE-2 low for both models, since exact two-word matches with one short sentence are rare.
