# Pop Culture & Meme Analysis

## Overview

This project explores how large language models (LLMs) can be used to extract, structure, and explain pop culture and meme-related knowledge from internet data.  
The original task was to scrape memes and local pop-cultural content directly from the web and generate text-only explanatory instructions describing the events, ideas, and entities referenced in those memes.

Due to practical limitations with live scraping, the project instead uses the MemeCap dataset (Reddit-based meme captions) as the primary meme source, combined with Wikipedia background retrieval, to construct a large-scale instruction-style supervised fine-tuning (SFT) dataset suitable for training or adapting LLMs.

The final output is a 3,000-entry prompt–completion dataset following Hugging Face formatting conventions.

This dataset was created for research and educational purposes.

---

## License

CC BY-NC-ND 4.0

This dataset was created for research and educational purposes.

---

## Problem Statement

Memes and pop culture references are highly contextual and often difficult for language models to interpret without external background knowledge.  

To address this, the project designs a pipeline that:
- Connects meme captions to background knowledge
- Extracts structured knowledge representations
- Iteratively refines that knowledge using LLM feedback
- Converts the result into instruction–completion training data

---

## Data Sources

- Meme data:  
  - MemeCap dataset (Reddit-based meme captions)
  
- Background knowledge:  
  - Wikipedia pages scraped for relevant entities, events, and concepts referenced by memes

All data is processed in English only.

---

## Method Overview

The solution is designed as a multi-stage NLP pipeline that combines reference grounding, background retrieval and LLM-based refinement.

### High-level steps

1. Data selection
   - Meme captions are sampled from the MemeCap dataset (title, image caption, meme caption)
   - Relevant Wikipedia pages are retrieved as background context
   
2. OCR of images
   - Meme images are run through an OCR to extract any text which are later filtered

3. Reference grounding
   - Relevant information is extracted from the MemeCap dataset data and image OCR outputs

4. Chunking
   - Wikipedia documents are chunked into smaller text segments for efficient processing

5. Indexing layer
   - Chunks are indexed and combined into a database

6. LLM explanation generation
   - The university LLM endpoint is used to generate explanations for the memes
   - If the response is unsatisfactory, a fixing step is implemented

7. Dataset assembly
   - Final LLM outputs are converted into instruction–completion examples
   - The dataset follows Hugging Face SFT conventions

---

## Code Usage:

1. memes_ocr.py
2. memecap_dataset_to_csv_jsonl.py
3. reference_grounding.py
4. build_raw_corpus.py
5. chunk_corpus.py
6. build_fts.py
7. build_faiss.py
8. retrieve.py
9. merge_meme_info.py
10. SFT_generation.py
11. SFT_dataset_creation.py

+ common.py

---

## Model Usage

All language understanding and generation steps are performed using a university-hosted LLM endpoint

The model is used for explanation and reasoning generation.

No external commercial APIs are required.

---

## Output

The final deliverable is:

-  A 3,000-entry supervised fine-tuning (SFT) dataset

- Format:
  - System and user prompt instruction
  - Model completion
  
- Purpose:
  - Training or adapting LLMs to better understand meme and pop culture references
  
The dataset has also been uploaded to HuggingFace, although it is currently in a private repository:
https://huggingface.co/datasets/marcell10651/meme-explanations-sft

---

## Evaluation

This project does not include formal quantitative evaluation metrics.  
The focus is on:
- Pipeline design
- Data quality
- Structural consistency of the generated knowledge

---

## Limitations

- Live scraping of memes was not implemented, the project relies on a pre-existing dataset
- No automatic accuracy or coverage metrics are computed
- Knowledge quality depends on LLM consistency and Wikipedia coverage
- Cultural interpretation is limited to English-language, Reddit memes

---

## Future Work

Possible extensions include:
- Direct scraping from multiple platforms
- Multilingual meme analysis
- Evaluation metrics for explanation quality
- Training and benchmarking models on the generated dataset

---

## Repository Contents

- 'data/' – Processed datasets and intermediate files  
- 'src/' – Scripts implementing pipeline stages  
- 'docs/' – One page project summary and pipeline flowchart
- 'README.md' – Project documentation

---

## Author

Individual university project by Marcell Bérces.
