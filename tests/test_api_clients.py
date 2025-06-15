"""
Unit tests for API client functions.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json

# Add the parent directory to the sys.path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.api_clients import (
    calculate_emission, get_emission_factors, estimate_shipping_emissions, 
    get_facility_compliance, get_company_details, query_claude, query_openai,
    _handle_response, APIError, RateLimitError
)

class TestApiClients(unittest.TestCase):
    """Test cases for API client functions."""
    
    def test_handle_response_success(self):
        """Test successful response handling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        
        result = _handle_response(mock_response, "Test API")
        self.assertEqual(result, {"result": "success"})
    
    def test_handle_response_rate_limit(self):
        """Test rate limit error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "rate limit exceeded"}
        mock_response.raise_for_status.side_effect = Exception("429 Client Error")
        
        with self.assertRaises(RateLimitError):
            _handle_response(mock_response, "Test API")
    
    def test_handle_response_api_error(self):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "bad request"}
        mock_response.raise_for_status.side_effect = Exception("400 Client Error")
        
        with self.assertRaises(APIError):
            _handle_response(mock_response, "Test API")
    
    @patch('app.utils.api_clients.requests.post')
    def test_calculate_emission(self, mock_post):
        """Test emission calculation with API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "co2e": 120.5,
            "co2e_unit": "kg",
            "co2e_calculation_method": "ar4",
            "co2e_calculation_origin": "source"
        }
        mock_post.return_value = mock_response
        
        # Call function
        result = calculate_emission("transportation", {"distance": 100, "distance_unit": "km"})
        
        # Assertions
        self.assertEqual(result["co2e"], 120.5)
        self.assertEqual(result["co2e_unit"], "kg")
    
    @patch('app.utils.api_clients.requests.get')
    def test_get_emission_factors(self, mock_get):
        """Test getting emission factors with API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "activity_id": "transportation_road_hgv",
                    "name": "Heavy Goods Vehicle (HGV)",
                    "category": "transport",
                    "unit": "km",
                    "co2e_factor": 0.12
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Call function
        result = get_emission_factors("transport")
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["activity_id"], "transportation_road_hgv")
    
    @patch('app.utils.api_clients.requests.post')
    def test_estimate_shipping_emissions(self, mock_post):
        """Test shipping emissions estimation with API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "id": "test-id",
                "type": "shipping_estimate",
                "attributes": {
                    "weight_value": 1000,
                    "weight_unit": "kg",
                    "distance_value": 500,
                    "distance_unit": "km",
                    "transport_method": "truck",
                    "carbon_g": 50000,
                    "carbon_lb": 110.23,
                    "carbon_kg": 50,
                    "carbon_mt": 0.05
                }
            }
        }
        mock_post.return_value = mock_response
        
        # Call function
        result = estimate_shipping_emissions(1000, 500, "truck")
        
        # Assertions
        self.assertEqual(result["data"]["attributes"]["carbon_kg"], 50)
        self.assertEqual(result["data"]["attributes"]["transport_method"], "truck")
    
    @patch('app.utils.api_clients.Anthropic')
    def test_query_claude(self, mock_anthropic):
        """Test querying Claude AI model."""
        # Setup mock response
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="AI response")]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client
        
        # Call function
        result = query_claude("Test prompt")
        
        # Assertions
        self.assertEqual(result, "AI response")
        mock_client.messages.create.assert_called_once()

if __name__ == '__main__':
    unittest.main()
