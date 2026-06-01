from typing import Optional, Dict, List
import re


def extract_gt_answer_gsm8k(text: str) -> float:
    """For preprocessing gsm8k dataset ground truth answer"""
    final_line = text.split('\n')[-1]
    assert final_line.startswith('#### ')
    final_line = final_line[5:].strip()

    numbers_only = re.sub(r'[^\d.]', '', final_line) # only digits
    return float(numbers_only)


def extract_gt_answer_aime(text: str) -> float:
    """For preprocessing aime dataset ground truth answer"""
    numbers_only = re.sub(r'[^\d.]', '', text) # only digits
    return float(numbers_only)


def get_final_box_match(text: str) -> Optional[str]:
    # Find all content inside \boxed{...}
    boxed_pattern = r'boxed\{([^}]*)\}'
    boxed_matches = re.findall(boxed_pattern, text, re.DOTALL)
    if not boxed_matches:
        return None

    # take final box answer
    return boxed_matches[-1].strip()


def extract_verifier_answer(text) -> Optional[bool]:
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    # first check the box
    if boxed_content.lower() in ['incorrect', 'correct']:
        return boxed_content.lower() == 'correct'

    # fall back to string matching in the whole text
    if ('wrong' in text.lower()) or ('incorrect' in text.lower()):
        return False
    elif 'correct' in text.lower():
        return True

    print(f"[WARNING] Failed extracting answer: {boxed_content}")
    return None


def extract_float_answer(text: str) -> Optional[float]:
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        numbers_only = re.sub(r'[^\d.]', '', boxed_content) # only digits
        return float(numbers_only)
    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None


def extract_sat_answer(text: str) -> Optional[Dict[str, bool]]:
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        # Parse to our best ability
        assignments = {}
        for line in boxed_content.split('\n'):
            var, value = line.strip().split()
            var, value = var.lower(), value.upper()
            if not (len(var) == 1 and var.isalpha() and value in ['T', 'F']):
                continue
            assignments[var] = (value == 'T')
        return assignments

    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None


def extract_sudoku_answer(text: str) -> Optional[List[List[int]]]:
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        # Parse to our best ability
        grid = []
        for line in boxed_content.split('\n'):
            clean_line = re.sub(r'\s+', '', line) # remove whitespace
            # Parse each character as a digit
            row = []
            for char in clean_line:
                if char.isdigit():
                    row.append(int(char))
            if len(row) > 0:  # Only add non-empty rows
                grid.append(row)

        return grid

    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None


def extract_matmul_answer(text: str) -> Optional[List[List[int]]]:
    boxed_content = get_final_box_match(text)
    if boxed_content == None:
        print(f"[WARNING] Failed extracting answer, no box.")
        return None

    try:
        # Parse to our best ability
        matrix = []
        for line in boxed_content.split('\n'):
            row = []
            for num_str in line.strip().split():
                if num_str.lstrip('-').isdigit():
                    row.append(int(num_str))
            if len(row) > 0:
                matrix.append(row)
        return matrix

    except:
        print(f"[WARNING] Failed extracting answer: {boxed_content}")
        return None
