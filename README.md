# Code Review Model

An AI-powered code review and correction system based on the CodeT5 model.

## Features

- Automatically identify and fix common coding errors
- Fix syntax issues in Python code
- Correct indentation problems
- Add missing colons, parentheses, and other syntax elements

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd contact

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Using the Pre-trained Model

```python
from src.models.code_correction import review_code

code_with_errors = """
def calculate_sum(a, b)
    return a + b
"""

corrected_code = review_code(code_with_errors)
print(corrected_code)
```

### Using the Fine-tuned Model

```python
python use_finetuned_model.py
```

### Training Your Own Model

```bash
python scripts/train_model.py
```

## Project Structure

```
contact/
├── src/                    # Source code directory
│   ├── data/               # Data processing utilities
│   ├── models/             # Model definitions
│   ├── training/           # Training utilities
│   └── utils/              # General utilities
├── notebooks/              # Jupyter notebooks for experimentation
├── examples/               # Example code
├── scripts/                # Scripts for various tasks
├── tests/                  # Unit tests
└── ...
```

## License

[MIT License](LICENSE)

python examples/basic_review.py
python scripts/train_model.py
python use_finetuned_model.py