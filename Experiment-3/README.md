# Experiment 3: Stemming, Lemmatization and Regular Expressions

**Name:** Rudransh Sogani  
**SAP ID:** 500120440  

## Objective
To implement and analyze core text normalization techniques (Stemming and Lemmatization) using the NLTK library, and to perform pattern-based information extraction from unstructured text using Regular Expressions (Regex).

## Workflow and Architecture

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontSize': '16px', 'primaryColor': '#ffffff', 'primaryTextColor': '#000000', 'primaryBorderColor': '#000000', 'lineColor': '#000000', 'secondaryColor': '#ffffff', 'tertiaryColor': '#ffffff' }}}%%
flowchart TD
    classDef whiteBox fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000,font-weight:bold,font-size:15px;
    classDef subBox fill:#ffffff,stroke:#222222,stroke-width:2px,stroke-dasharray: 4 4,color:#000000,font-weight:bold,font-size:16px;

    subgraph StemmingStage["1. STEMMING WORKFLOW (Porter Stemmer)"]
        S1["Input Word (e.g., 'studies', 'playing')"]:::whiteBox
        S2["Apply Algorithmic Suffix Stripping Rules"]:::whiteBox
        S3["Heuristic Character Truncation (No Dictionary Lookup)"]:::whiteBox
        S4["Generated Stem Output: 'studi', 'play'"]:::whiteBox
        S1 --> S2 --> S3 --> S4
    end

    subgraph LemmatizationStage["2. LEMMATIZATION WORKFLOW (WordNet)"]
        L1["Input Word (e.g., 'went', 'better', 'mice')"]:::whiteBox
        L2["Assign Contextual Part-of-Speech Tag (POS: Verb, Adj, Noun)"]:::whiteBox
        L3["Morphological and Vocabulary Analysis in WordNet Database"]:::whiteBox
        L4["Canonical Dictionary Lemma Output: 'go', 'good', 'mouse'"]:::whiteBox
        L1 --> L2 --> L3 --> L4
    end

    subgraph RegexStage["3. REGULAR EXPRESSION EXTRACTION WORKFLOW"]
        R1["Raw Unstructured Text Corpus"]:::whiteBox
        R2["Define Modular Regex Patterns (Emails, URLs, Phones, Tags)"]:::whiteBox
        R3["Pattern Matching Execution via re.findall()"]:::whiteBox
        R4["Isolated Structured Entities Output"]:::whiteBox
        R1 --> R2 --> R3 --> R4
    end

    S4 --> L1
    L4 --> R1

    class StemmingStage,LemmatizationStage,RegexStage subBox;
```

## Procedure & Implementation
The experiment is structured into three dedicated sections:

1. **Stemming (`Experiment 3.1`):**
   * Initializing `nltk.stem.PorterStemmer` <sup>(`PorterStemmer()`)</sup>.
   * Passing the target word corpus: `playing`, `played`, `plays`, `studies`, `studying`, `connected`, `connection`, `computers`.
   * Applying rule-based suffix stripping to extract the morphological root (stem) <sup>(`stemmer.stem(word)`)</sup>.
   * Displaying the original words mapped against their corresponding stems.

2. **Lemmatization (`Experiment 3.2`):**
   * Initializing `nltk.stem.WordNetLemmatizer` with WordNet corpus support <sup>(`WordNetLemmatizer()`)</sup>.
   * Processing input words: `cats`, `dogs`, `running`, `runs`, `ran`, `studies`, `studying`, `better`, `children`, `mice`, `went`, `ate`, `leaves`, `caring`.
   * Incorporating Part-of-Speech (POS) tags to guide morphological resolution for nouns, verbs, and adjectives <sup>(`lemmatizer.lemmatize(word, pos)`)</sup>.
   * Producing canonical dictionary base forms (lemmas) for irregular inflections and plurals.

3. **Information Extraction with Regular Expressions (`Experiment 3.3`):**
   * Defining modular regex patterns for distinct entity classes <sup>(`import re`)</sup>:
     * **Email addresses:** `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`
     * **URLs:** `https?://[^\s]+|\bwww\.[^\s]+\b`
     * **Mobile numbers:** `(?:\+91[-\s]?)?[6-9]\d{9}`
     * **Hashtags:** `#[A-Za-z0-9_]+`
     * **Mentions:** `(?<!\S)@[A-Za-z0-9_]+`
   * Executing extraction across unstructured workshop text using `re.findall()` <sup>(`re.findall(pattern, text)`)</sup>.
   * Isolating and formatting extracted entities.

## Observations

| Experiment Stage | Technique / Module | Target Input | Result / Transformed Output |
| :--- | :--- | :--- | :--- |
| **Stemming** | `nltk.stem.PorterStemmer` | Inflected and affixed words (`studies`, `computers`) | Truncated base forms (`studi`, `comput`) |
| **Lemmatization (Noun)** | `WordNetLemmatizer` (POS: `n`) | Plural and irregular nouns (`children`, `mice`, `leaves`) | Canonical singular forms (`child`, `mouse`, `leaf`) |
| **Lemmatization (Verb)** | `WordNetLemmatizer` (POS: `v`) | Irregular past/continuous verbs (`went`, `ran`, `ate`) | Root infinitive verbs (`go`, `run`, `eat`) |
| **Lemmatization (Adj)** | `WordNetLemmatizer` (POS: `a`) | Comparative adjective (`better`) | Base positive adjective (`good`) |
| **Regex: Emails** | `re` pattern matching | Unstructured text | `nlpworkshop@gmail.com`, `support@python.org` |
| **Regex: URLs** | `re` pattern matching | Unstructured text | `https://www.nlpworkshop.com`, `www.python.org`, `https://github.com/NLPWorkshop` |
| **Regex: Mobile Numbers**| `re` pattern matching | Unstructured text | `+91-9876543210`, `9123456789` |
| **Regex: Social Tags** | `re` pattern matching | Unstructured text | `#NLP`, `#Python`, `#MachineLearning`, `@NLPWorkshop`, `@PythonLearner` |

## Result Interpretation & Learnings
* **Heuristic vs. Morphological Normalization:** Stemming operates on algorithmic suffix stripping without vocabulary verification, often yielding non-dictionary tokens (such as `studi`). In contrast, Lemmatization leverages morphological vocabularies to guarantee valid lexical headwords.
* **Contextual POS Significance:** Lemmatization requires explicit or inferred Part-of-Speech context to accurately resolve irregular verb conjugations (such as `went` to `go`) and comparative adjectives (such as `better` to `good`).
* **Entity Extraction Precision:** Regular expressions provide a deterministic and robust approach for parsing unstructured text to isolate structured contact details, web links, and social metadata.
