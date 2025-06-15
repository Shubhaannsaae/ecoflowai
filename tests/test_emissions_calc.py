"""
Unit tests for emissions calculation functions.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to the sys.path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.emissions_calc import (
    calculate_logistics_emissions, calculate_procurement_emissions,
    calculate_supplier_emissions, calculate_total_emissions,
    generate_optimization_targets, EMISSION_FACTORS
)

class TestEmissionsCalc(unittest.TestCase):
    """Test cases for emissions calculation functions."""
    
    def setUp(self):
        """Set up test data."""
        # Sample logistics data
        self.logistics_data = pd.DataFrame({
            'shipment_id': ['S001', 'S002', 'S003'],
            'transport_mode': ['Truck', 'Ship', 'Air'],
            'distance_km': [500, 2000, 1000],
            'weight_kg': [1000, 5000, 200]
        })
        
        # Sample procurement data
        self.procurement_data = pd.DataFrame({
            'order_id': ['P001', 'P002', 'P003'],
            'product_category': ['Electronics', 'Packaging', 'Raw Materials'],
            'quantity': [10, 100, 50],
            'weight_kg': [2, 0.5, 10],
            'unit_cost': [500, 5, 20]
        })
        
        # Sample supplier data
        self.supplier_data = pd.DataFrame({
            'supplier_id': ['SUP001', 'SUP002', 'SUP003'],
            'supplier_name': ['Supplier A', 'Supplier B', 'Supplier C'],
            'category': ['Electronics', 'Packaging', 'Raw Materials'],
            'country': ['USA', 'China', 'Germany'],
            'annual_revenue_usd': [1000000, 5000000, 2000000],
            'sustainability_certification': ['ISO 14001', 'None', 'B Corp']
        })
    
    def test_calculate_logistics_emissions(self):
        """Test logistics emissions calculation."""
        # Calculate emissions
        result_df, summary = calculate_logistics_emissions(self.logistics_data)
        
        # Assertions
        self.assertEqual(len(result_df), 3)
        self.assertIn('emissions_kg_co2e', result_df.columns)
        self.assertIn('total_emissions_kg', summary)
        self.assertIn('emissions_by_mode', summary)
        
        # Check if all modes have emissions
        for mode in self.logistics_data['transport_mode'].unique():
            self.assertIn(mode, summary['emissions_by_mode'])
        
        # Verify that air has higher emissions per tonne-km than ship
        air_row = result_df[result_df['transport_mode'] == 'Air']
        ship_row = result_df[result_df['transport_mode'] == 'Ship']
        
        self.assertGreater(
            air_row['emissions_g_co2e_per_tonne_km'].values[0],
            ship_row['emissions_g_co2e_per_tonne_km'].values[0]
        )
    
    def test_calculate_procurement_emissions(self):
        """Test procurement emissions calculation."""
        # Calculate emissions
        result_df, summary = calculate_procurement_emissions(self.procurement_data)
        
        # Assertions
        self.assertEqual(len(result_df), 3)
        self.assertIn('emissions_kg_co2e', result_df.columns)
        self.assertIn('emissions_kg_co2e_per_unit', result_df.columns)
        self.assertIn('total_emissions_kg', summary)
        self.assertIn('emissions_by_category', summary)
        
        # Check if all categories have emissions
        for category in self.procurement_data['product_category'].unique():
            self.assertIn(category, summary['emissions_by_category'])
    
    def test_calculate_supplier_emissions(self):
        """Test supplier emissions calculation."""
        # Calculate emissions
        result_df, summary = calculate_supplier_emissions(self.supplier_data)
        
        # Assertions
        self.assertEqual(len(result_df), 3)
        self.assertIn('carbon_footprint_kg', result_df.columns)
        self.assertIn('sustainability_score', result_df.columns)
        self.assertIn('certification_score', result_df.columns)
        self.assertIn('total_supplier_footprint_kg', summary)
        
        # Check certification scores - ISO 14001 should have a higher score than None
        iso_certified = result_df[result_df['sustainability_certification'] == 'ISO 14001']
        non_certified = result_df[result_df['sustainability_certification'] == 'None']
        
        self.assertGreater(
            iso_certified['certification_score'].values[0],
            non_certified['certification_score'].values[0]
        )
    
    def test_calculate_total_emissions(self):
        """Test total emissions calculation."""
        # Calculate individual emissions first
        logistics_result = calculate_logistics_emissions(self.logistics_data)
        procurement_result = calculate_procurement_emissions(self.procurement_data)
        supplier_result = calculate_supplier_emissions(self.supplier_data)
        
        # Calculate total emissions
        total_emissions = calculate_total_emissions(
            logistics_summary=logistics_result[1],
            procurement_summary=procurement_result[1],
            supplier_summary=supplier_result[1]
        )
        
        # Assertions
        self.assertIn('total_emissions_kg', total_emissions)
        self.assertIn('total_emissions_tonnes', total_emissions)
        self.assertIn('emissions_by_scope', total_emissions)
        self.assertIn('emissions_by_category', total_emissions)
        
        # Check scopes
        self.assertIn('scope1', total_emissions['emissions_by_scope'])
        self.assertIn('scope2', total_emissions['emissions_by_scope'])
        self.assertIn('scope3', total_emissions['emissions_by_scope'])
        
        # Ensure total is sum of components
        expected_total = (
            logistics_result[1]['total_emissions_kg'] +
            procurement_result[1]['total_emissions_kg']
        )
        
        # Allow small difference due to rounding
        self.assertAlmostEqual(
            total_emissions['total_emissions_kg'],
            expected_total,
            delta=0.1  # Allow 0.1 kg difference
        )
    
    def test_generate_optimization_targets(self):
        """Test generation of optimization targets."""
        # Calculate emissions first
        logistics_result = calculate_logistics_emissions(self.logistics_data)
        procurement_result = calculate_procurement_emissions(self.procurement_data)
        
        # Calculate total emissions
        total_emissions = calculate_total_emissions(
            logistics_summary=logistics_result[1],
            procurement_summary=procurement_result[1]
        )
        
        # Generate optimization targets
        optimization_targets = generate_optimization_targets(
            total_summary=total_emissions,
            logistics_df=logistics_result[0],
            procurement_df=procurement_result[0]
        )
        
        # Assertions
        self.assertIn('targets', optimization_targets)
        self.assertIn('recommendations', optimization_targets)
        self.assertIn('total_potential_reduction_kg', optimization_targets)
        self.assertIn('potential_reduction_percent', optimization_targets)
        
        # Check target structure
        self.assertIn('overall', optimization_targets['targets'])
        
        # Check recommendation structure
        self.assertTrue(len(optimization_targets['recommendations']) > 0)
        first_rec = optimization_targets['recommendations'][0]
        self.assertIn('category', first_rec)
        self.assertIn('title', first_rec)
        self.assertIn('description', first_rec)
        self.assertIn('potential_reduction_kg', first_rec)
        
        # Ensure potential reduction is positive and less than total emissions
        self.assertGreater(optimization_targets['total_potential_reduction_kg'], 0)
        self.assertLess(
            optimization_targets['total_potential_reduction_kg'],
            total_emissions['total_emissions_kg']
        )

if __name__ == '__main__':
    unittest.main()
