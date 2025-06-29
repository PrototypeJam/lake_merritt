# Bluebook Citation Evaluation Packs

This directory contains two evaluation packs for assessing legal citations according to The Bluebook citation format:

## 1. bluebook_eval.yaml
- **Purpose**: Semantic evaluation of legal citations using LLM judgment
- **Scorer**: `llm_judge` with GPT-4
- **Threshold**: 0.5 (configurable)
- **Features**:
  - Uses custom prompt template for citation-specific evaluation
  - Provides detailed reasoning for each score
  - Rubric:
    - 1.0 = Substantively identical citations
    - 0.5 = Same authority but minor formatting issues
    - 0.0 = Different or incorrect authority

## 2. bluebook_normalized_eval.yaml
- **Purpose**: Two-stage evaluation combining normalized matching with LLM judgment
- **Scorers**: 
  1. `normalized_exact_match` - Tolerant string matching
  2. `llm_judge` - Semantic evaluation
- **Features**:
  - First stage ignores trailing punctuation differences
  - Second stage provides semantic analysis
  - Useful for distinguishing between cosmetic and substantive differences

## Sample Data
See `examples/data/bluebook_citations.csv` for a sample dataset with 16 legal citation test cases.

## Usage Example
```bash
# Run the semantic evaluation
python -m streamlit run streamlit_app.py -- \
  --eval-pack examples/eval_packs/bluebook_eval.yaml \
  --data examples/data/bluebook_citations.csv

# Run the two-stage evaluation
python -m streamlit run streamlit_app.py -- \
  --eval-pack examples/eval_packs/bluebook_normalized_eval.yaml \
  --data examples/data/bluebook_citations.csv
```

## Expected Results
- **bluebook_eval.yaml**: Will catch substantive differences while being lenient on minor formatting
- **bluebook_normalized_eval.yaml**: 
  - Stage 1 (normalized_match): Expected to pass 12-14 out of 16 cases
  - Stage 2 (semantic_judge): Provides reasoning for all cases