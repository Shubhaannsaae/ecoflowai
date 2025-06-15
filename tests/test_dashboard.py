"""
Unit tests for dashboard components.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
import streamlit as st

# Add the parent directory to the sys.path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.dashboard.components import (
    render_kpi_metrics, create_emissions_summary_chart,
    create_optimization_impact_chart, create_supplier_sustainability_chart
)

class TestDashboardComponents(unittest.TestCase):
    """Test cases for dashboard components."""
    
    def setUp(self):
        """Set up test data."""
        # Sample emissions data
        self.emissions_data = {
            'total_emissions_kg': 1500,
            'total_emissions_tonnes': 1.5,
            'emissions_by_scope': {
                'scope1': 100,
                'scope2': 200,
                'scope3': 1200
            },
            'emissions_by_category': {
                'Logistics': 800,
                'Procurement': 500,
                'Supplier': 200
            },
            'scope_percentages': {
                'scope1': 6.67,
                'scope2': 13.33,
                'scope3': 80.0
            }
        }
        
        # Sample optimization data
        self.optimization_data = {
            'targets': {
                'overall': {
                    'current_emissions_kg': 1500,
                    'reduction_target_percent': 10,
                    'reduction_target_kg': 150,
                    'target_emissions_kg': 1350
                }
            },
            'recommendations': [
                {
                    'category': 'Logistics',
                    'title': 'Shift Air Freight to Sea',
                    'description': 'Shifting air freight to sea transport can reduce emissions by 80%',
                    'potential_reduction_kg': 100,
                    'difficulty': 'Medium',
                    'timeframe': 'Medium-term'
                },
                {
                    'category': 'Procurement',
                    'title': 'Source Recycled Packaging',
                    'description': 'Using recycled packaging materials reduces emissions',
                    'potential_reduction_kg': 50,
                    'difficulty': 'Low',
                    'timeframe': 'Short-term'
                }
            ],
            'total_potential_reduction_kg': 150,
            'potential_reduction_percent': 10
        }
        
        # Sample supplier data
        self.supplier_data = pd.DataFrame({
            'supplier_id': ['SUP001', 'SUP002', 'SUP003', 'SUP004'],
            'supplier_name': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D'],
            'sustainability_score': [8.5, 4.2, 7.8, 3.5],
            'sustainability_risk': [1.5, 5.8, 2.2, 6.5],
            'category': ['Electronics', 'Packaging', 'Raw Materials', 'Chemicals']
        })
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_emissions_summary_chart(self, mock_close, mock_savefig, mock_figure):
        """Test emissions summary chart creation."""
        # Mock BytesIO
        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = b'test_image_data'
        
        with patch('io.BytesIO', return_value=mock_buffer):
            # Create chart
            chart_data = create_emissions_summary_chart(self.emissions_data)
        
        # Assertions
        self.assertIsNotNone(chart_data)
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_optimization_impact_chart(self, mock_close, mock_savefig, mock_figure):
        """Test optimization impact chart creation."""
        # Mock BytesIO
        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = b'test_image_data'
        
        with patch('io.BytesIO', return_value=mock_buffer):
            # Create chart
            chart_data = create_optimization_impact_chart(self.optimization_data)
        
        # Assertions
        self.assertIsNotNone(chart_data)
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_supplier_sustainability_chart(self, mock_close, mock_savefig, mock_figure):
        """Test supplier sustainability chart creation."""
        # Mock BytesIO
        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = b'test_image_data'
        
        with patch('io.BytesIO', return_value=mock_buffer):
            # Create chart
            chart_data = create_supplier_sustainability_chart(self.supplier_data)
        
        # Assertions
        self.assertIsNotNone(chart_data)
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_create_supplier_sustainability_chart_empty_data(self):
        """Test supplier sustainability chart with empty data."""
        # Create empty dataframe
        empty_df = pd.DataFrame()
        
        # Create chart
        chart_data = create_supplier_sustainability_chart(empty_df)
        
        # Assertions
        self.assertIsNone(chart_data)
    
    def test_create_emissions_summary_chart_empty_data(self):
        """Test emissions summary chart with empty data."""
        # Create empty emissions data
        empty_data = {}
        
        # Create chart
        chart_data = create_emissions_summary_chart(empty_data)
        
        # Assertions
        self.assertIsNone(chart_data)

if __name__ == '__main__':
    unittest.main()
