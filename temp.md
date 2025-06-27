# International Trade Model Analysis: Caliendo & Parro (2015) Implementation

## Project Overview
This project implements and visualizes a Caliendo & Parro (2015) international trade model with a focus on counterfactual tariff policy analysis. The system solves for general equilibrium solutions and provides interactive visualization of trade network effects.

## Core Architecture

### 1. Model Management (`models.py`)
Manages all model variables through three main classes:
- **ModelParams**: Handles model parameters calibrated from real-world datasets (loaded from CSV files)
- **ModelShock**: Manages structural model shocks and policy interventions
- **ModelSol**: Stores solution variables and equilibrium outcomes

### 2. Equation System (`equations.py`)
Contains all mathematical equations that define the model's system of equations for general equilibrium computation.

### 3. Solver Engine (`solvers.py`)
Implements optimization algorithms to solve for the model's general equilibrium, ensuring computational efficiency and numerical stability.

### 4. Visualization Interface (`app.py`)
Provides interactive visualization using Streamlit framework for exploring counterfactual analysis results.

## Network Visualization Specifications

### Equilibrium Representation
The visualization focuses on "hat algebra" equilibrium analysis, comparing changes between:
- **Model 1**: Benchmark scenario (real-world tariff structure)
- **Model 2**: Counterfactual scenario (policy intervention)

### Network Structure
- **Nodes**: Each node represents a sector within a specific country
- **Clusters**: Country-level groupings of sector nodes
- **Edges**: Trade volume connections between sectors (N×S by N×S matrix, where N=countries, S=sectors)

### Visual Encoding Rules

#### Edge Representation
- **Thickness**: Proportional to trade volume
- **Directional encoding**: Edge thickness at node A represents exports from A to destination node
- **Bidirectional flows**: Both import and export volumes are visualized simultaneously

#### Node Size Encoding
- **Sector nodes**: Size represents sector-specific variable magnitudes (N×S matrices)
- **Cluster nodes**: Large overlay nodes represent country-level aggregates (N-dimensional vectors)

#### Color Encoding (Hat Algebra Changes)
- **Red spectrum**: Positive changes (increases from baseline)
  - Darker red = larger increases
- **Green spectrum**: Negative changes (decreases from baseline)
  - Darker green = larger decreases
- **Neutral**: Minimal or no change from baseline

## Development Priorities

### Code Quality Improvements
1. **Computational Efficiency**: Optimize numerical algorithms without altering computational logic
2. **Code Structure**: Enhance modularity and Pythonic design patterns
3. **Documentation**: Improve code readability and maintainability
4. **Preservation**: Maintain existing computational accuracy and model behavior

### Visualization Enhancements
1. **Interactive Network**: Implement dynamic trade network visualization
2. **Comparative Analysis**: Enable side-by-side comparison of equilibrium states
3. **User Interface**: Streamlined controls for policy scenario exploration
4. **Performance**: Efficient rendering for large-scale trade networks

## Technical Constraints
- Preserve existing computational logic and numerical results
- Maintain compatibility with current data input formats
- Ensure backward compatibility with existing model calibrations