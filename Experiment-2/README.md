# Experiment 2: Basic Text Preprocessing

**Name:** Rudransh Sogani  
**SAP ID:** 500120440  

## Objective
To implement foundational text preprocessing techniques essential for Natural Language Processing by transforming raw textual data into structured tokens while removing noise.

## Procedure & Implementation
The experiment operates on raw text files and is divided into three core preprocessing stages, leveraging built-in Python string operations, NLTK, and spaCy libraries.

1.  **Text Cleaning (`2.1_text_data.txt`):**
    *   Reading the raw text file <sup>(`open(file, 'r').read()`)</sup>.
    *   Calculating the frequency of uppercase characters <sup>(`sum(c.isupper())`)</sup>.
    *   Converting the entire text corpus to lowercase <sup>(`str.lower()`)</sup> for uniformity.
    *   Identifying and stripping all punctuation marks using the `string.punctuation` module <sup>(`c not in string.punctuation`)</sup>.
    *   Removing numeric digits <sup>(`str.isdigit()`)</sup>.
    *   Normalizing whitespace by splitting and rejoining the text <sup>(`" ".join(str.split())`)</sup>, calculating the total whitespace characters removed.

2.  **Tokenization (`2.2_tokenization_data.txt`):**
    *   **NLTK:** Utilizing `nltk.word_tokenize` and `nltk.sent_tokenize` for word and sentence level tokenization <sup>(`list`)</sup>.
    *   **spaCy:** Loading the `en_core_web_sm` model <sup>(`spacy.load()`)</sup> to process the text and extract word and sentence tokens via the document object <sup>(`[t.text for t in doc]`)</sup>.
    *   **Python Native:** Implementing basic tokenization using the `split()` method for words and regular expressions (`re.split`) <sup>(`re.split(r'(?<=[.!?]) +', text)`)</sup> for sentence boundary detection.

3.  **Stop Words Removal (`2.3_clean_data.txt`):**
    *   Downloading and importing the English stop words list from `nltk.corpus` <sup>(`set(stopwords.words('english'))`)</sup>.
    *   Tokenizing the lowercase input text.
    *   Filtering the token list to exclude any words present in the stop words set <sup>(`t not in stop_words`)</sup> and ensuring only alphanumeric tokens remain <sup>(`str.isalnum()`)</sup>.
    *   Isolating and displaying the specific stop words that were removed from the corpus.

## Observations
| Preprocessing Stage | Method / Library | Action Performed | Result Output |
| :--- | :--- | :--- | :--- |
| **Text Cleaning** | Built-in Python (`string`) | Lowercasing, punctuation/digit removal | Normalized, noise-free text string |
| **Whitespace Normalization**| Built-in Python (`split`, `join`) | Elimination of redundant spaces | Compressed text corpus |
| **Tokenization (NLTK)** | `nltk.word_tokenize`, `sent_tokenize`| Rule-based token extraction | Lists of discrete words and sentences |
| **Tokenization (spaCy)** | `en_core_web_sm` model | Model-based document parsing | Extracted tokens via `.text` attribute |
| **Tokenization (Native)** | `str.split()`, `re.split()` | Whitespace and regex boundaries | Basic token lists |
| **Stop Word Removal** | `nltk.corpus.stopwords` | Filtering against standard English list | Alphanumeric tokens with core semantic value |

## Result Interpretation & Learnings
*   **Pipeline Foundation:** The experiment practically demonstrates the critical initial phases necessary for natural language processing pipelines.
*   **Effective Normalization:** The text cleaning stage successfully normalizes raw text by eliminating non-informative characters, including punctuation, digits, and excess whitespace.
*   **Tokenizer Comparison:** The tokenization phase highlights syntactical differences and operational efficiencies between rule-based approaches (NLTK, regex) and model-based architectures (spaCy).
*   **Dimensionality Reduction:** The stop word removal process effectively reduces the dimensionality of the text data corpus.
*   **Semantic Isolation:** Post preprocessing, only the semantically significant tokens remain, perfectly formatted for downstream computational linguistics tasks.
