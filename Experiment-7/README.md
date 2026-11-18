# Experiment 7: Sentiment Analysis on IMDB Movie Reviews

**Name:** Rudransh Sogani  
**SAP ID:** 500120440  

## Objective
To implement sentiment analysis on the IMDB Movie Reviews dataset using Logistic Regression, LSTM, a Vanilla Transformer, and the pretrained BERT model, and to perform a comparative analysis of all the models using Precision, Recall, F1-score, and the Confusion Matrix.

## Procedure & Implementation
This experiment involves multiple phases covering a classical baseline, two neural models trained from scratch, a pretrained model, and their comparison:

1.  **Dataset & Preprocessing:**
    *   Downloading the IMDB 50K reviews dataset (`IMDB-Dataset.csv`) from the sample-code GitHub repo <sup>(`urllib.request.urlretrieve()`, `pd.read_csv()`)</sup>.
    *   Taking a balanced chunk of 10,000 reviews (5,000 positive + 5,000 negative) instead of the full dataset <sup>(`groupby('sentiment').sample(5000)`)</sup>.
    *   Cleaning the text by removing `<br />` tags and non-letter characters, then lowercasing <sup>(`re.sub()`, `str.lower()`)</sup>.
    *   Encoding labels (positive = 1, negative = 0) and creating a stratified split of 8,000 train / 2,000 test reviews <sup>(`train_test_split(stratify=...)`)</sup>.
    *   Defining a common evaluation function that prints the classification report and plots the confusion matrix <sup>(`classification_report()`, `confusion_matrix()`, `sns.heatmap()`)</sup>.

2.  **Logistic Regression (`7.1`):**
    *   Converting reviews to TF-IDF features with unigrams and bigrams and English stopwords removed <sup>(`TfidfVectorizer(max_features=50000, ngram_range=(1, 2))`)</sup>.
    *   Training a logistic regression classifier <sup>(`LogisticRegression(C=5, max_iter=1000)`)</sup>.
    *   Extracting the most positive and most negative words from the learned weights <sup>(`log_reg.coef_`, `get_feature_names_out()`)</sup>.

3.  **Sequence Preparation for LSTM and Transformer:**
    *   Building a 20,000-word vocabulary from the training reviews <sup>(`Counter.most_common()`)</sup>.
    *   Converting each review to a fixed-length sequence of 250 word IDs, with padding and unknown-word handling <sup>(`to_sequence()`, `torch.tensor()`)</sup>.
    *   Writing a shared training loop with Adam, binary cross-entropy loss, and gradient clipping <sup>(`BCEWithLogitsLoss()`, `clip_grad_norm_()`)</sup>.

4.  **LSTM (`7.2`):**
    *   Building an embedding layer (128-d) followed by a single-layer LSTM (128 hidden units) <sup>(`nn.Embedding`, `nn.LSTM`)</sup>.
    *   Classifying from the final hidden state through dropout and a linear layer <sup>(`hidden[-1]`, `nn.Dropout(0.3)`, `nn.Linear`)</sup>.
    *   Training for 6 epochs (~2.7M parameters).

5.  **Vanilla Transformer (`7.3`):**
    *   Building an encoder-only Transformer with token embeddings plus learned positional embeddings <sup>(`nn.Embedding(max_len, d_model)`)</sup>.
    *   Stacking 2 self-attention encoder layers with 4 heads and padding masks <sup>(`nn.TransformerEncoderLayer`, `src_key_padding_mask`)</sup>.
    *   Mean-pooling the non-padded token outputs before the classification layer.
    *   Training for 6 epochs at learning rate 5e-4 (~2.9M parameters).

6.  **BERT (`7.4`):**
    *   Loading pretrained `bert-base-uncased` with a classification head <sup>(`AutoTokenizer`, `AutoModelForSequenceClassification`)</sup>.
    *   Tokenizing reviews with BERT's WordPiece tokenizer, truncated/padded to 256 tokens <sup>(`bert_tokenizer(truncation=True, max_length=256)`)</sup>.
    *   Fine-tuning on a balanced subset of 2,000 training reviews for 3 epochs, due to laptop hardware limits <sup>(`AdamW(lr=2e-5)`, `get_linear_schedule_with_warmup()`)</sup>.
    *   Evaluating on the same 2,000 test reviews as the other models <sup>(`logits.argmax(1)`)</sup>.

7.  **Comparative Analysis (`7.5`):**
    *   Tabulating Accuracy, Precision, Recall, and F1 for all four models <sup>(`pd.DataFrame(results)`)</sup>.
    *   Plotting a grouped bar chart of the metrics and all four confusion matrices side by side <sup>(`DataFrame.plot(kind='bar')`, `plt.subplots(1, 4)`)</sup>.
    *   Identifying the best model by F1 and counting false positives and false negatives for each model <sup>(`idxmax()`, `cm[0, 1]`, `cm[1, 0]`)</sup>.

## Observations
| Model | Accuracy | Precision | Recall | F1 | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8850 | 0.8738 | 0.9000 | 0.8867 | 130 | 100 |
| **LSTM** | 0.7655 | 0.7465 | 0.8040 | 0.7742 | 273 | 196 |
| **Vanilla Transformer** | 0.7950 | 0.8300 | 0.7420 | 0.7835 | 152 | 258 |
| **BERT** | **0.9100** | **0.8846** | **0.9430** | **0.9129** | **123** | **57** |

| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **Logistic Regression** | Top positive words: `great`, `excellent`, `love`, `perfect`; top negative: `worst`, `bad`, `awful`, `poor` | TF-IDF weights, confusion matrix |
| **LSTM** | Lowest scores of all models; most false positives (273) | Training loss/accuracy per epoch, confusion matrix |
| **Vanilla Transformer** | Higher precision (0.83) but lowest recall (0.742); most false negatives (258) | Training loss/accuracy per epoch, confusion matrix |
| **BERT** | Training accuracy 0.77 → 0.92 → 0.97 over 3 epochs (~62 s/epoch); best F1 (0.9129) | Fine-tuning log, confusion matrix |
| **Comparison** | Ranking by F1: BERT > Logistic Regression > Transformer > LSTM | Grouped bar chart, side-by-side confusion matrices |

## Result Interpretation & Learnings
*   **Pretraining Wins:** BERT achieved the best Accuracy (91.0%), Recall (94.3%), and F1 (0.913) despite being fine-tuned on only 2,000 reviews, because it brings language knowledge learned from large-scale pretraining.
*   **Strong Classical Baseline:** TF-IDF + Logistic Regression came very close to BERT (F1 0.887) while training in seconds, showing that sentiment is largely carried by individual words and short phrases like *"waste"* or *"excellent"*.
*   **Data-Hungry Neural Models:** The LSTM and Transformer, trained from scratch on just 8,000 reviews, scored well below Logistic Regression, as they must learn word meanings and structure from very little data.
*   **Error Patterns:** The LSTM leaned towards predicting *positive* (more false positives), while the Transformer leaned towards *negative* (more false negatives, recall 0.742), which the confusion matrices make clear even where F1 scores are similar.
*   **Long Reviews:** With an average length of ~229 words, truncating to 250 (LSTM/Transformer) or 256 tokens (BERT) cuts some reviews short, and a unidirectional LSTM relying on its last hidden state struggles to keep information from early in the review.
*   **Accuracy vs. Cost:** BERT gives the best results but needs a GPU and much more time and memory, so Logistic Regression remains a practical choice when resources are limited.
