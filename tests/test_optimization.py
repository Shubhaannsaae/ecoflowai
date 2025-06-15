"""
Unit tests for optimization functions.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add the parent directory to the sys.path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.optimization import (
    generate_optimal_routes, generate_sustainable_procurement_plan,
    generate_supplier_sustainability_plan, generate_ai_sustainability_strategy
)

class TestOptimization(unittest.TestCase):
    """Test cases for optimization functions."""
    
    def setUp(self):
        """Set up test data."""
        # Sample logistics data with emissions
        self.logistics_data = pd.DataFrame({
            'shipment_id': ['S001', 'S002', 'S003', 'S004'],
            'transport_mode': ['Air', 'Truck', 'Ship', 'Air'],
            'distance_km': [1500, 800, 3000, 2000],
            'weight_kg': [500, 1000, 5000, 300],
            'emissions_kg_co2e': [600, 100, 150, 700],
            'emissions_g_co2e_per_tonne_km': [800, 125, 10, 1167],
            'date': pd.to_datetime(['2023-01-15', '2023-01-20', '2023-02-10', '2023-01-18']),
            'origin': ['New York', 'Chicago', 'Shanghai', 'London'],
            'destination': ['London', 'Miami', 'Rotterdam', 'Paris']
        })
        
        # Sample procurement data with emissions
        self.procurement_data = pd.DataFrame({
            'order_id': ['P001', 'P002', 'P003', 'P004'],
            'product_category': ['Electronics', 'Packaging', 'Raw Materials', 'Chemicals'],
            'quantity': [10, 100, 50, 25],
            'weight_kg': [2, 0.5, 10, 5],
            'unit_cost': [500, 5, 20, 30],
            'emissions_kg_co2e': [400, 50, 500, 375],
            'emissions_kg_co2e_per_unit': [40, 0.5, 10, 15]
        })
        
        # Sample supplier data with sustainability scores
        self.supplier_data = pd.DataFrame({
            'supplier_id': ['SUP001', 'SUP002', 'SUP003', 'SUP004'],
            'supplier_name': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D'],
            'category': ['Electronics', 'Packaging', 'Raw Materials', 'Chemicals'],
            'country': ['USA', 'China', 'Germany', 'India'],
            'sustainability_score': [8.5, 4.2, 7.8, 3.5],
            'sustainability_risk': [1.5, 5.8, 2.2, 6.5],
            'annual_revenue_usd': [1000000, 5000000, 2000000, 800000]
        })
    
    def test_generate_optimal_routes(self):
        """Test optimal route generation."""
        # Calculate optimal routes
        optimized_df, summary = generate_optimal_routes(self.logistics_data)
        
        # Assertions
        self.assertEqual(len(optimized_df), len(self.logistics_data))
        self.assertIn('optimized_transport_mode', optimized_df.columns)
        self.assertIn('optimized_emissions_kg_co2e', optimized_df.columns)
        self.assertIn('emissions_reduction_kg_co2e', optimized_df.columns)
        self.assertIn('optimization_method', optimized_df.columns)
        
        # Check summary
        self.assertIn('original_emissions_kg', summary)
        self.assertIn('optimized_emissions_kg', summary)
        self.assertIn('total_reduction_kg', summary)
        self.assertIn('percent_reduction', summary)
        self.assertIn('reduction_by_method', summary)
        
        # Ensure optimized emissions are lower than original
        self.assertLess(summary['optimized_emissions_kg'], summary['original_emissions_kg'])
        
        # Check if at least some shipments were optimized
        optimized_count = len(optimized_df[optimized_df['optimization_method'] != 'No change'])
        self.assertGreater(optimized_count, 0)
        
        # Check if air freight has been optimized to other modes
        air_shipments = self.logistics_data[self.logistics_data['transport_mode'] == 'Air']
        optimized_air = optimized_df.loc[air_shipments.index]
        
        mode_changed = False
        for idx, row in optimized_air.iterrows():
            if row['optimized_transport_mode'] != 'Air':
                mode_changed = True
                break
        
        self.assertTrue(mode_changed, "Expected at least one air shipment to be optimized to a different mode")
    
    def test_generate_sustainable_procurement_plan(self):
        """Test sustainable procurement plan generation."""
        # Generate sustainable procurement plan
        optimized_df, summary = generate_sustainable_procurement_plan(self.procurement_data)
        
        # Assertions
        self.assertEqual(len(optimized_df), len(self.procurement_data))
        self.assertIn('sustainable_alternative', optimized_df.columns)
        self.assertIn('emissions_reduction_percent', optimized_df.columns)
        self.assertIn('optimized_emissions_kg_co2e', optimized_df.columns)
        self.assertIn('emissions_reduction_kg_co2e', optimized_df.columns)
        
        # Check summary
        self.assertIn('original_emissions_kg', summary)
        self.assertIn('optimized_emissions_kg', summary)
        self.assertIn('total_reduction_kg', summary)
        self.assertIn('percent_reduction', summary)
        self.assertIn('reduction_by_category', summary)
        
        # Ensure optimized emissions are lower than original
        self.assertLess(summary['optimized_emissions_kg'], summary['original_emissions_kg'])
        
        # Check if all categories have alternatives
        for category in self.procurement_data['product_category'].unique():
            category_rows = optimized_df[optimized_df['product_category'] == category]
            self.assertNotEqual(
                'No alternative identified',
                category_rows['sustainable_alternative'].iloc[0],
                f"No sustainable alternative identified for category {category}"
            )
    
    def test_generate_supplier_sustainability_plan(self):
        """Test supplier sustainability plan generation."""
        # Generate supplier sustainability plan
        optimized_df, summary = generate_supplier_sustainability_plan(self.supplier_data)
        
        # Assertions
        self.assertEqual(len(optimized_df), len(self.supplier_data))
        self.assertIn('recommendation', optimized_df.columns)
        self.assertIn('priority', optimized_df.columns)
        self.assertIn('potential_improvement', optimized_df.columns)
        self.assertIn('engagement_strategy', optimized_df.columns)
        
        # Check summary
        self.assertIn('current_avg_sustainability_score', summary)
        self.assertIn('potential_avg_sustainability_score', summary)
        self.assertIn('improvement_potential', summary)
        self.assertIn('high_priority_supplier_count', summary)
        
        # Ensure potential score is higher than current
        self.assertGreater(
            summary['potential_avg_sustainability_score'],
            summary['current_avg_sustainability_score']
        )
        
        # Check if high-risk suppliers have high priority
        high_risk_suppliers = self.supplier_data[self.supplier_data['sustainability_score'] < 5]
        high_priority_count = len(optimized_df.loc[high_risk_suppliers.index][optimized_df['priority'] == 'High'])
        self.assertEqual(high_priority_count, len(high_risk_suppliers))
    
    @patch('app.utils.optimization.query_claude')
    def test_generate_ai_sustainability_strategy(self, mock_query):
        """Test AI sustainability strategy generation."""
        # Mock Claude response
        mock_query.return_value = """
# Sustainable Supply Chain Strategy

## Executive Summary
This is a test strategy summary.

## Strategic Recommendations
1. Test recommendation 1
2. Test recommendation 2

## Implementation Roadmap
- Month 1-3: Phase 1
- Month 4-6: Phase 2

## Business Case
Sustainability investments will yield long-term benefits.

## Key Performance Indicators
- KPI 1
- KPI 2
"""
        
        # Generate AI strategy
        strategy = generate_ai_sustainability_strategy(
            total_emissions={'total_emissions_kg': 1000},
            optimization_targets={'total_potential_reduction_kg': 200}
        )
        
        # Assertions
        self.assertIn('strategy_text', strategy)
        self.assertIn('timestamp', strategy)
        self.assertIn('generated_by', strategy)
        
        # Check strategy content
        self.assertIn('Executive Summary', strategy['strategy_text'])
        self.assertIn('Strategic Recommendations', strategy['strategy_text'])
        self.assertIn('Implementation Roadmap', strategy['strategy_text'])
        
        # Verify Claude was called
        mock_query.assert_called_once()

if __name__ == '__main__':
    unittest.main()
