# Experiment 6: GloVe Word Embeddings and N-gram Language Model

**Name:** Rudransh Sogani  
**SAP ID:** 500120440  

## Objective
To implement the GloVe (Global Vectors for Word Representation) algorithm to generate word embeddings from a text corpus and find the words most similar to *"king"*, both from scratch and using a pretrained Gensim model, and to implement a statistical N-gram language model from scratch that predicts the next word in a sequence based on the preceding words.

## Procedure & Implementation
This experiment involves multiple phases covering word embeddings and statistical language modelling:

1.  **GloVe from Scratch (`6.1a`):**
    *   Reading the corpus line by line from the dataset (`glove_corpus.txt`) <sup>(`open()`, `str.strip()`)</sup>.
    *   Lowercasing, tokenizing, and removing common function words <sup>(`re.findall(r"[a-z]+")`, custom stopword set)</sup>.
    *   Building the vocabulary and word-to-index mappings <sup>(`Counter()`, `word2idx`, `idx2word`)</sup>.
    *   Constructing a distance-weighted co-occurrence matrix with a context window of 3 <sup>(`np.zeros((V, V))`, weight `1/d`)</sup>.
    *   Training word and context vectors with the GloVe weighted least-squares objective <sup>(`f(x) = (x/x_max)^0.75`, `log X_ij`)</sup>.
    *   Optimizing with AdaGrad for 1000 epochs and plotting the training loss <sup>(per-parameter gradient accumulators, `plt.plot()`)</sup>.
    *   Combining main and context vectors into the final embeddings <sup>(`W + Wc`)</sup>.
    *   Finding the words most similar to *"king"* <sup>(cosine similarity)</sup>.
    *   Visualizing selected word embeddings in 2-D <sup>(`PCA(n_components=2)`)</sup>.

2.  **GloVe with Pretrained Model (`6.1b`):**
    *   Loading pretrained 100-dimensional GloVe vectors trained on Wikipedia + Gigaword <sup>(`gensim.downloader.load('glove-wiki-gigaword-100')`)</sup>.
    *   Finding the top 10 words most similar to *"king"* <sup>(`glove.most_similar()`)</sup>.
    *   Solving the word analogy *king − man + woman* <sup>(`most_similar(positive=[...], negative=[...])`)</sup>.
    *   Comparing pairwise similarity scores, e.g. *king–queen* vs. *king–cat* <sup>(`glove.similarity()`)</sup>.
    *   Listing nearest neighbours for other words from the corpus (*queen, computer, cat, sun*).

3.  **N-gram Language Model from Scratch (`6.2`):**
    *   Tokenizing the given 8-sentence NLP corpus and padding each sentence with `<s>` and `</s>` tokens.
    *   Implementing a reusable `NGramModel` class that counts context → next-word occurrences <sup>(`defaultdict(Counter)`)</sup>.
    *   Estimating conditional probabilities with add-k smoothing <sup>(`(C(context, w) + k) / (C(context) + k|V|)`, `k = 0.01`)</sup>.
    *   Training unigram, bigram, and trigram models on the same corpus.
    *   Predicting the top 3 next words for test phrases such as *"natural language"* and *"deep learning is a"* <sup>(`predict_next()`)</sup>.
    *   Generating sentences greedily from seed words until `</s>` is predicted <sup>(`generate()`)</sup>.
    *   Evaluating each model on training and unseen test sentences <sup>(`perplexity()`)</sup>.

## Observations
| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **Co-occurrence Matrix** | 61-word vocabulary, 220 non-zero word pairs | Distance-weighted counts (window = 3) |
| **GloVe (From Scratch)** | Nearest to *king*: `son`, `queen`, `wise`, `powerful`, `prince` | Cosine similarity, loss curve, PCA plot |
| **GloVe (Pretrained)** | Neighbours like `prince`, `queen`, `monarch`, `throne`; *king − man + woman ≈ queen* | Gensim `glove-wiki-gigaword-100` |
| **N-gram Prediction** | *"natural language"* → `processing` (0.90 with trigram) | Bigram vs. trigram top-3 probabilities |
| **Sentence Generation** | *"a language"* → *"a language model predicts the next word in a sentence"* | Greedy trigram decoding |
| **Perplexity** | Train PPL: unigram ≈ 26.6, bigram ≈ 2.07, trigram ≈ 1.75 | Add-k smoothed probability evaluation |

## Result Interpretation & Learnings
*   **Global Statistics:** GloVe learns embeddings from corpus-wide co-occurrence counts rather than individual context windows, so words sharing contexts (*king*, *queen*, *prince*) end up close in vector space.
*   **Weighting Function:** The weighting function `f(x)` stops very frequent pairs from dominating training while still giving rare pairs some influence.
*   **Pretrained vs. Scratch:** The 24-sentence corpus only captures local associations (*king* ↔ *son*, *powerful*), while pretrained vectors trained on billions of tokens capture real semantic relations and solve analogies like *king − man + woman ≈ queen*.
*   **Context Length in N-grams:** Higher-order models use more preceding words, giving sharper predictions (*"deep learning is a"* → `branch` / `part`) and lower training perplexity than unigram or bigram models.
*   **Overfitting Trade-off:** The trigram model scores best on training text but worse than the bigram on unseen sentences (test PPL ≈ 2.39 vs. 1.95), showing that longer contexts become sparse on small corpora.
*   **Smoothing:** Add-k smoothing gives unseen word sequences a small non-zero probability, so the model never assigns zero likelihood to a valid but unseen sentence.
