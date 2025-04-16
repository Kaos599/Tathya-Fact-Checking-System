"""
Tests for the agents module.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import pytest

from fact_check_system.schemas import AtomicClaim, VerificationResult
from fact_check_system.agents import (
    extract_claims_agent,
    decompose_claims_agent,
    primary_verification_agent,
    cross_verification_agent,
    adjudication_agent,
    synthesis_agent
)

class TestAgents(unittest.TestCase):
    """Test cases for agent functions."""
    
    def test_extract_claims_agent(self):
        """Test extract_claims_agent function."""
        # Mock the LLM chain invoke method
        with patch('langchain_core.output_parsers.StrOutputParser.invoke') as mock_invoke:
            mock_invoke.return_value = "The Earth orbits the Sun.\nParis is the capital of France."
            
            # Call the function
            result = extract_claims_agent("Here are some facts: The Earth orbits the Sun. Paris is the capital of France.")
            
            # Assert the result
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], "The Earth orbits the Sun.")
            self.assertEqual(result[1], "Paris is the capital of France.")
    
    def test_decompose_claims_agent(self):
        """Test decompose_claims_agent function."""
        # Mock the LLM chain invoke method
        with patch('langchain.output_parsers.PydanticOutputParser.invoke') as mock_invoke:
            mock_invoke.return_value = [
                AtomicClaim(
                    id="1234",
                    statement="The Earth orbits the Sun",
                    entities=["Earth", "Sun"],
                    time_references=[],
                    location_references=[],
                    numeric_values=[],
                    importance=1.0
                )
            ]
            
            # Call the function
            result = decompose_claims_agent("The Earth orbits the Sun")
            
            # Assert the result
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].statement, "The Earth orbits the Sun")
            self.assertEqual(result[0].entities, ["Earth", "Sun"])
    
    def test_primary_verification_agent(self):
        """Test primary_verification_agent function."""
        # Create a test claim
        claim = AtomicClaim(
            id="1234",
            statement="The Earth orbits the Sun",
            entities=["Earth", "Sun"],
            time_references=[],
            location_references=[],
            numeric_values=[],
            importance=1.0
        )
        
        # Create test evidence
        evidence_collection = {
            "search_results": [
                {
                    "id": "e1",
                    "source": "web",
                    "url": "https://example.com/astronomy",
                    "content": "The Earth orbits the Sun once every 365.25 days.",
                    "credibility_score": 0.9,
                    "relevance_score": 0.95
                }
            ],
            "knowledge_base": [],
            "news_sources": []
        }
        
        # Mock the LLM chain invoke method
        with patch('langchain.output_parsers.PydanticOutputParser.invoke') as mock_invoke:
            mock_invoke.return_value = VerificationResult(
                agent_id="primary",
                verdict="TRUE",
                confidence=0.95,
                reasoning="The evidence clearly supports that the Earth orbits the Sun.",
                evidence_assessment={"e1": 0.95},
                reasoning_difficulty=0.1,
                claim_ambiguity=0.1
            )
            
            # Call the function
            result = primary_verification_agent(claim, evidence_collection)
            
            # Assert the result
            self.assertEqual(result.agent_id, "primary")
            self.assertEqual(result.verdict, "TRUE")
            self.assertAlmostEqual(result.confidence, 0.95)
            self.assertEqual(result.reasoning, "The evidence clearly supports that the Earth orbits the Sun.")
    
    def test_cross_verification_agent(self):
        """Test cross_verification_agent function."""
        # Similar to test_primary_verification_agent but with secondary LLM
        # Create a test claim
        claim = AtomicClaim(
            id="1234",
            statement="The Earth orbits the Sun",
            entities=["Earth", "Sun"],
            time_references=[],
            location_references=[],
            numeric_values=[],
            importance=1.0
        )
        
        # Create test evidence
        evidence_collection = {
            "search_results": [
                {
                    "id": "e1",
                    "source": "web",
                    "url": "https://example.com/astronomy",
                    "content": "The Earth orbits the Sun once every 365.25 days.",
                    "credibility_score": 0.9,
                    "relevance_score": 0.95
                }
            ],
            "knowledge_base": [],
            "news_sources": []
        }
        
        # Mock the LLM chain invoke method
        with patch('langchain.output_parsers.PydanticOutputParser.invoke') as mock_invoke:
            mock_invoke.return_value = VerificationResult(
                agent_id="cross",
                verdict="TRUE",
                confidence=0.9,
                reasoning="Independent verification confirms the Earth orbits the Sun.",
                evidence_assessment={"e1": 0.9},
                reasoning_difficulty=0.1,
                claim_ambiguity=0.1
            )
            
            # Call the function
            result = cross_verification_agent(claim, evidence_collection)
            
            # Assert the result
            self.assertEqual(result.agent_id, "cross")
            self.assertEqual(result.verdict, "TRUE")
            self.assertAlmostEqual(result.confidence, 0.9)
            self.assertEqual(result.reasoning, "Independent verification confirms the Earth orbits the Sun.")
    
    # Add more tests for other agent functions as needed

if __name__ == '__main__':
    unittest.main() 