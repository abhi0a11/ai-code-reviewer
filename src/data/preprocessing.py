"""
Preprocessing utilities for code data
"""

def clean_code(code_text):
    """
    Clean the code text by removing unnecessary whitespace and normalizing line endings.
    
    Args:
        code_text: The code text to clean
        
    Returns:
        str: Cleaned code text
    """
    if not code_text:
        return ""
    
    # Normalize line endings
    code_text = code_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove trailing whitespace from each line
    lines = code_text.split('\n')
    lines = [line.rstrip() for line in lines]
    
    # Remove leading and trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)

def tokenize_code(code_text, tokenizer):
    """
    Tokenize code using the provided tokenizer.
    
    Args:
        code_text: The code text to tokenize
        tokenizer: The tokenizer to use
        
    Returns:
        dict: Tokenized code
    """
    cleaned_code = clean_code(code_text)
    return tokenizer(cleaned_code, return_tensors="pt", padding=True, truncation=True)

def extract_python_syntax_features(code_text):
    """
    Extract basic Python syntax features from code text.
    
    Args:
        code_text: The code text to analyze
        
    Returns:
        dict: Dictionary of syntax features
    """
    features = {
        "has_missing_colons": False,
        "has_indentation_issues": False,
        "has_mismatched_parentheses": False,
        "has_syntax_errors": False
    }
    
    lines = clean_code(code_text).split('\n')
    for i, line in enumerate(lines):
        # Check for missing colons
        if any(keyword in line for keyword in ["def ", "class ", "if ", "elif ", "else:", "for ", "while ", "try:", "except ", "finally:"]):
            if not line.strip().endswith(':'):
                features["has_missing_colons"] = True
        
        # Simple indentation check (very basic)
        if i > 0 and lines[i-1].strip().endswith(':') and line.strip() and not line.startswith(' '):
            features["has_indentation_issues"] = True
        
        # Very basic parentheses check
        if line.count('(') != line.count(')'):
            features["has_mismatched_parentheses"] = True
    
    return features
