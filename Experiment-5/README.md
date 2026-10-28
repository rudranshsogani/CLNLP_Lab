# Experiment 5: Subword Tokenization and POS Tagging

**Name:** Rudransh Sogani  
**SAP ID:** 500120440  

## Objective
To implement subword-level tokenization using Byte Pair Encoding (BPE) and SentencePiece tokenizers, with both pretrained and from-scratch approaches, and to perform Part-of-Speech (POS) tagging on textual data using spaCy and NLTK, including frequency analysis of tagged tokens.

## Procedure & Implementation
This experiment involves multiple phases covering subword tokenization and grammatical tagging:

1.  **BPE Tokenization (`5.1`):**
    *   Reading the first line of the dataset (`input_sub_word_data.txt`) as the sample sentence <sup>(`open()`, `str.split('\n')[0]`)</sup>.
    *   **a. Pretrained:** Tokenizing the sentence with GPT-2's BPE tokenizer <sup>(`GPT2Tokenizer.from_pretrained('gpt2')`)</sup>.
    *   Encoding the text and mapping IDs back to subword tokens <sup>(`tokenizer.encode()`, `convert_ids_to_tokens()`)</sup>.
    *   **b. From Scratch:** Training a new BPE model on the full corpus <sup>(`Tokenizer(BPE(unk_token='[UNK]'))`, `BpeTrainer(vocab_size=1000)`)</sup>.
    *   Splitting text on whitespace before learning merges <sup>(`Whitespace()` pre-tokenizer)</sup>.
    *   Displaying each generated token alongside its token ID <sup>(`encoding.tokens`, `encoding.ids`)</sup>.

2.  **SentencePiece Tokenization (`5.2`):**
    *   **a. Pretrained:** Tokenizing the same sentence with T5's SentencePiece tokenizer <sup>(`T5Tokenizer.from_pretrained('t5-small')`)</sup>.
    *   Displaying subword pieces (with the `▁` word-boundary marker) and their IDs <sup>(`convert_ids_to_tokens()`)</sup>.
    *   **b. From Scratch:** Training a SentencePiece model directly on the raw corpus <sup>(`spm.SentencePieceTrainer.train(vocab_size=500)`)</sup>.
    *   Automatically retrying with the largest vocabulary size the small corpus supports <sup>(`re.search()` on the trainer error)</sup>.
    *   Generating the model and vocabulary files (`m_sp.model`, `m_sp.vocab`) and encoding the sentence <sup>(`sp.encode_as_pieces()`, `sp.encode_as_ids()`)</sup>.

3.  **POS Tagging – Short Sentence (`5.3`):**
    *   Tagging the sentence *"The young student is reading an interesting book in the library."*
    *   **a. spaCy:** Processing with the small English pipeline <sup>(`spacy.load('en_core_web_sm')`, `token.pos_`)</sup>.
    *   Displaying each token, its POS tag, and the tag description <sup>(`spacy.explain()`)</sup>.
    *   **b. NLTK:** Tokenizing and tagging with the Penn Treebank tagset <sup>(`nltk.word_tokenize()`, `nltk.pos_tag()`)</sup>.
    *   Downloading both legacy and renamed NLTK resources for version compatibility <sup>(`nltk.download()`)</sup>.

4.  **POS Tagging with Frequency – Large File (`5.4`):**
    *   Processing the full corpus (`input_sub_word_data.txt`, first 5000 characters) with spaCy <sup>(`nlp(text[:5000])`)</sup>.
    *   Counting the frequency of each unique `(Token, POS Tag, Description)` combination, ignoring whitespace <sup>(`Counter()`, `token.is_space`)</sup>.
    *   Displaying the top 20 tagged tokens with their frequencies <sup>(`Counter.most_common(20)`)</sup>.

## Observations
| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **BPE (Pretrained)** | Subword tokens with `Ġ` space marker and GPT-2 vocabulary IDs | GPT-2 Tokenizer |
| **BPE (From Scratch)** | Corpus-specific merges and a custom 1000-size vocabulary | HuggingFace `tokenizers` BPE Trainer |
| **SentencePiece (Pretrained)** | Subword pieces with `▁` boundary marker and `</s>` end token | T5 Tokenizer |
| **SentencePiece (From Scratch)** | Learned pieces like `ization`, `tion`, `ing`, `▁un`, `▁re` | SentencePiece Trainer (`m_sp.model`, `m_sp.vocab`) |
| **POS Tagging (Sentence)** | Universal tags (spaCy) vs. Penn Treebank tags (NLTK) | Tabular Token / Tag / Description output |
| **POS Tagging (Frequency)** | Most frequent token–tag pairs in the corpus | `Counter` Frequency Ranking |

## Result Interpretation & Learnings
*   **Subword Mechanics:** BPE and SentencePiece split words into reusable units, letting the tokenizer handle rare and unseen words (e.g. *internationalization* → `▁international` + `ization`) without an oversized vocabulary.
*   **Pretrained vs. Scratch:** Pretrained tokenizers carry large general-purpose vocabularies, while scratch-trained models learn only the patterns in the given corpus, so their IDs and segmentations differ for the same sentence.
*   **Corpus Size Matters:** The small training corpus capped the achievable SentencePiece vocabulary size, showing why a representative, larger corpus produces more useful subword vocabularies.
*   **Tokenizer Conventions:** GPT-2 marks leading spaces with `Ġ`, whereas SentencePiece treats whitespace as a symbol (`▁`), allowing it to work directly on raw text without pre-tokenization.
*   **Tagset Comparison:** spaCy's coarse Universal POS tags are easier to read, while NLTK's fine-grained Penn Treebank tags encode extra detail such as tense and number.
*   **Linguistic Profiling:** POS frequency analysis reveals the grammatical makeup of a corpus, with determiners, nouns, and prepositions dominating technical prose.
