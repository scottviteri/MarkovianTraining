"""
Unit tests for answer extraction functions in evaluation.py.

Tests extraction functions with edge cases to ensure robust answer parsing:
- MCQ: Word boundary extraction vs legacy extraction
- Numeric: Anchor method, simple method, and normalization
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from evaluation import (
    extract_letter,
    extract_letter_word_boundary,
    extract_answer_simple,
    extract_answer_with_anchor,
    extract_answer,
)


class TestMCQExtraction:
    """Test multiple choice question extraction methods."""
    
    def test_word_boundary_correct_extraction(self):
        """Test word boundary extraction with isolated letters."""
        assert extract_letter_word_boundary("The answer is B") == "B"
        assert extract_letter_word_boundary("I choose A") == "A"
        assert extract_letter_word_boundary("Answer: C") == "C"
        assert extract_letter_word_boundary("Option D is correct") == "D"
        assert extract_letter_word_boundary("E") == "E"
    
    def test_word_boundary_avoids_false_matches(self):
        """Test word boundary extraction avoids matching letters inside words."""
        # These should NOT match letters inside words
        assert extract_letter_word_boundary("The best choice") == "X"  # No isolated letter
        assert extract_letter_word_boundary("Select the answer") == "X"  # 'e' in Select/the
        assert extract_letter_word_boundary("Please choose wisely") == "X"  # 'e' in Please
        assert extract_letter_word_boundary("Decide carefully") == "X"  # 'D' in Decide, 'e' in Decide
    
    def test_word_boundary_with_punctuation(self):
        """Test word boundary extraction with punctuation."""
        assert extract_letter_word_boundary("Answer: B.") == "B"
        assert extract_letter_word_boundary("(C)") == "C"
        assert extract_letter_word_boundary("A.") == "A"
        assert extract_letter_word_boundary("The answer is B, which...") == "B"
    
    def test_word_boundary_case_insensitive(self):
        """Test word boundary extraction is case insensitive."""
        assert extract_letter_word_boundary("answer: b") == "B"
        assert extract_letter_word_boundary("ANSWER: C") == "C"
        assert extract_letter_word_boundary("Answer: d") == "D"
    
    def test_word_boundary_first_match(self):
        """Test word boundary extraction returns first isolated letter."""
        assert extract_letter_word_boundary("A or B") == "A"  # First isolated letter
        assert extract_letter_word_boundary("not A but B") == "A"
        assert extract_letter_word_boundary("Between C and D, choose C") == "C"
    
    def test_legacy_extraction_issues(self):
        """Test legacy extraction has known false positive issues."""
        # Legacy method will incorrectly match letters inside words
        assert extract_letter("The answer is D") == "E"  # Matches 'E' in "The"
        assert extract_letter("Select B") == "E"  # Matches 'E' in "Select"
        assert extract_letter("Please choose C") == "E"  # Matches 'E' in "Please"
    
    def test_legacy_extraction_correct_cases(self):
        """Test legacy extraction works when letter appears early."""
        assert extract_letter("A is correct") == "A"
        assert extract_letter("B is the answer") == "B"
        assert extract_letter("C") == "C"


class TestNumericExtractionSimple:
    """Test simple numeric extraction method."""
    
    def test_simple_with_equals(self):
        """Test simple extraction with equals sign."""
        assert extract_answer_simple("x = 42") == 42
        assert extract_answer_simple("The answer is 10 = 10") == 10
        assert extract_answer_simple("5 + 5 = 10") == 10
    
    def test_simple_without_equals(self):
        """Test simple extraction without equals sign."""
        assert extract_answer_simple("The answer is 42") == 42
        assert extract_answer_simple("42 apples") == 42
        assert extract_answer_simple("Total: 100") == 100
    
    def test_simple_with_comma(self):
        """Test simple extraction removes commas."""
        assert extract_answer_simple("1,000") == 1000
        assert extract_answer_simple("The answer is 1,234,567") == 1234567
    
    def test_simple_negative_numbers(self):
        """Test simple extraction handles negative numbers."""
        assert extract_answer_simple("-42") == -42
        assert extract_answer_simple("The result is -10") == -10
    
    def test_simple_invalid_cases(self):
        """Test simple extraction returns [invalid] for non-numeric text."""
        assert extract_answer_simple("No numbers here") == "[invalid]"
        assert extract_answer_simple("") == "[invalid]"
        assert extract_answer_simple("abc") == "[invalid]"


class TestNumericExtractionAnchor:
    """Test anchor-based numeric extraction method."""
    
    def test_anchor_with_answer_label(self):
        """Test anchor extraction prioritizes 'Answer:' label."""
        assert extract_answer_with_anchor("First 10, then Answer: 42") == 42
        assert extract_answer_with_anchor("x = 5, Answer: 10") == 10
        assert extract_answer_with_anchor("Answer: 42") == 42
    
    def test_anchor_case_insensitive_label(self):
        """Test anchor extraction is case insensitive for 'Answer' label."""
        assert extract_answer_with_anchor("answer: 42") == 42
        assert extract_answer_with_anchor("ANSWER: 42") == 42
        assert extract_answer_with_anchor("Answer 42") == 42  # Optional colon
    
    def test_anchor_with_equals_fallback(self):
        """Test anchor extraction falls back to equals sign if no Answer label."""
        assert extract_answer_with_anchor("x = 42") == 42
        assert extract_answer_with_anchor("5 + 5 = 10") == 10
    
    def test_anchor_with_first_number_fallback(self):
        """Test anchor extraction falls back to first number if no label or equals."""
        assert extract_answer_with_anchor("The result is 42") == 42
        assert extract_answer_with_anchor("42 apples total") == 42
    
    def test_anchor_priority_order(self):
        """Test anchor extraction priority: Answer > = > first number."""
        # Answer label should win even if other numbers present
        assert extract_answer_with_anchor("10 = 10, Answer: 42") == 42
        # Equals should win over earlier numbers
        assert extract_answer_with_anchor("First 5, then 3 + 2 = 10") == 10
    
    def test_anchor_with_comma(self):
        """Test anchor extraction removes commas."""
        assert extract_answer_with_anchor("Answer: 1,000") == 1000
        assert extract_answer_with_anchor("= 1,234") == 1234
    
    def test_anchor_negative_numbers(self):
        """Test anchor extraction handles negative numbers."""
        assert extract_answer_with_anchor("Answer: -42") == -42
        assert extract_answer_with_anchor("= -10") == -10
    
    def test_anchor_invalid_cases(self):
        """Test anchor extraction returns [invalid] for non-numeric text."""
        assert extract_answer_with_anchor("No numbers here") == "[invalid]"
        assert extract_answer_with_anchor("") == "[invalid]"
        assert extract_answer_with_anchor("Answer: unknown") == "[invalid]"


class TestNumericExtractionUnified:
    """Test unified extract_answer function with method selection."""
    
    def test_method_simple(self):
        """Test extract_answer with 'simple' method."""
        assert extract_answer("x = 42", method="simple") == 42
        assert extract_answer("First 10, then Answer: 42", method="simple") == 10  # No priority
    
    def test_method_anchor(self):
        """Test extract_answer with 'anchor' method."""
        assert extract_answer("First 10, then Answer: 42", method="anchor") == 42  # Has priority
        assert extract_answer("x = 42", method="anchor") == 42
    
    def test_method_invalid(self):
        """Test extract_answer raises error for invalid method."""
        with pytest.raises(ValueError, match="Unknown extraction method"):
            extract_answer("42", method="invalid_method")


class TestMCQEdgeCases:
    """Test edge cases for MCQ extraction."""
    
    def test_parentheses_format(self):
        """Test extraction with parentheses format."""
        assert extract_letter_word_boundary("(B)") == "B"
        assert extract_letter_word_boundary("(A) is correct") == "A"
    
    def test_dash_format(self):
        """Test extraction with dash format."""
        assert extract_letter_word_boundary("- B") == "B"
        assert extract_letter_word_boundary("A -") == "A"
    
    def test_multiple_isolated_letters(self):
        """Test extraction returns first isolated letter when multiple present."""
        assert extract_letter_word_boundary("A or B") == "A"
        assert extract_letter_word_boundary("not C but D") == "C"
    
    def test_no_valid_letter(self):
        """Test extraction returns 'X' when no valid letter found."""
        assert extract_letter_word_boundary("The best option") == "X"
        assert extract_letter_word_boundary("123") == "X"
        assert extract_letter_word_boundary("") == "X"
    
    def test_out_of_range_letters(self):
        """Test extraction ignores letters outside valid range."""
        # For A-E range (5 choices)
        assert extract_letter_word_boundary("F is not valid") == "X"
        assert extract_letter_word_boundary("Z") == "X"


class TestNumericEdgeCases:
    """Test edge cases for numeric extraction."""
    
    def test_multiple_numbers(self):
        """Test extraction with multiple numbers."""
        # Simple method: first after equals or first overall
        assert extract_answer_simple("5 + 10 = 15") == 15
        # Anchor method: prioritizes Answer label
        assert extract_answer_with_anchor("10 + 5 = 15, Answer: 100") == 100
    
    def test_decimal_numbers(self):
        """Test extraction with decimal numbers (extracts integer part)."""
        # Current implementation extracts first integer, so 42 from 42.5
        assert extract_answer_simple("42.5") == 42
        assert extract_answer_with_anchor("Answer: 42.5") == 42
    
    def test_whitespace_handling(self):
        """Test extraction handles various whitespace."""
        assert extract_answer_with_anchor("  Answer:   42  ") == 42
        assert extract_answer_simple("  =   10  ") == 10
    
    def test_special_characters(self):
        """Test extraction ignores special characters."""
        assert extract_answer_simple("$42") == 42
        assert extract_answer_with_anchor("Answer: #100") == 100
    
    def test_zero(self):
        """Test extraction handles zero correctly."""
        assert extract_answer_simple("0") == 0
        assert extract_answer_with_anchor("Answer: 0") == 0


class TestExtractionRealExamples:
    """Test extraction with real model outputs."""
    
    def test_mcq_real_outputs(self):
        """Test MCQ extraction with realistic model outputs."""
        # Good formats
        assert extract_letter_word_boundary(
            "Let me analyze each option. After careful consideration, the answer is B."
        ) == "B"
        
        assert extract_letter_word_boundary(
            "Looking at the choices:\nA) Incorrect\nB) Incorrect\nC) Correct\nD) Incorrect\nAnswer: C"
        ) == "C"
        
        # Should avoid false matches
        assert extract_letter_word_boundary(
            "The best approach here is to eliminate options systematically."
        ) == "X"  # 'e' in "The" shouldn't match
    
    def test_numeric_real_outputs(self):
        """Test numeric extraction with realistic model outputs."""
        # Good format with Answer label
        result = extract_answer_with_anchor(
            "Let me solve this step by step.\n"
            "First, 5 + 3 = 8\n"
            "Then, 8 × 2 = 16\n"
            "Finally, 16 - 4 = 12\n"
            "Answer: 12"
        )
        assert result == 12
        
        # Format without clear Answer label
        result = extract_answer_with_anchor(
            "Working through the problem:\n"
            "x = 10\n"
            "y = 5\n"
            "x + y = 15"
        )
        assert result == 15
        
        # Ambiguous case (anchor method prioritizes Answer label)
        result = extract_answer_with_anchor(
            "Step 1: Calculate 10 + 5 = 15\n"
            "Step 2: Multiply by 2 to get 30\n"
            "Answer: 30"
        )
        assert result == 30  # Should get 30, not 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

