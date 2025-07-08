# Streamlit Visualization Framework

## Overview

This document describes the comprehensive visualization framework for the economic model analysis application built with Streamlit. The framework provides interactive visualization, real-time model solving, and data export capabilities with support for both local and API-based architectures.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Performance Analysis](#performance-analysis)
- [Core Components](#core-components)
- [User Interface Design](#user-interface-design)
- [Data Flow](#data-flow)
- [Performance Optimization](#performance-optimization)
- [Caching Strategy](#caching-strategy)
- [Deployment Considerations](#deployment-considerations)

## Architecture Overview

The visualization framework follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Application Layer                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   app.py        │  │ visualization.py │  │ model_pipeline.py │             │
│  │ Main Interface  │  │ UI Components   │  │ Model Solving    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Model Computation Layer                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   models.py     │  │  equations.py   │  │   solvers.py    │             │
│  │ Data Structures │  │ Math Framework  │  │ Algorithms      │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Data Layer                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   data.npz      │  │  Session State  │  │  Cache Layer    │             │
│  │ Model Data      │  │ User State      │  │ Performance     │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Modular Architecture**: Clean separation between UI, computation, and data layers
2. **Caching Strategy**: Multiple levels of caching for optimal performance
3. **Session Management**: Persistent user state across interactions
4. **Dual Mode Support**: Both local and API-based operation modes
5. **Responsive Design**: Adaptive UI that handles large datasets efficiently

## Performance Analysis

### Identified Performance Bottlenecks

#### 1. **App Startup Lag (Primary Issue)**
**Root Cause**: Baseline model is solved immediately upon app startup
```python
# In main() function - lines 647-652
if st.session_state['baseline_solution'] is None:
    baseline_sol, baseline_params = solve_benchmark_unified()  # ⚠️ BLOCKING CALL
    st.session_state['baseline_solution'] = baseline_sol
    st.session_state['baseline_params'] = baseline_params
```

**Impact**: 
- 10-30 second delay on first app load
- "Running" indicator appears immediately
- User cannot interact with app during solving

**Current Mitigation**: Session state caching (prevents re-solving within session)

#### 2. **Variable Selection Lag (Previously Fixed)**
**Previous Issue**: Variable selection triggered model re-solving
**Solution**: Implemented session state caching for baseline model
**Status**: ✅ Resolved

#### 3. **Visualization Rendering Performance**
**Issues**:
- Large datasets (49 countries × 26 sectors) create heavy DOM
- Multiple Plotly charts rendered simultaneously
- Session state keys not optimized for variable-specific caching

#### 4. **Memory Usage**
**Issues**:
- Multiple model solutions stored in session state
- Large NumPy arrays duplicated across cache layers
- Streamlit cache + session state + Python objects

### Performance Metrics

| Operation | Cold Start | Warm Start | Memory Usage |
|-----------|------------|------------|--------------|
| App Startup | 15-30s | 1-2s | 200-500MB |
| Variable Selection | 0.1s | 0.1s | +50MB |
| Visualization | 2-5s | 1-2s | +100MB |
| Model Solving | 10-25s | 0.1s | +200MB |

## Core Components

### 1. Main Application (`app.py`)

**Purpose**: Primary Streamlit interface and orchestration
**Key Features**:
- Dual mode support (Local/API)
- Session state management
- Model type selection (Baseline/Counterfactual)
- Download functionality (Excel/CSV)

**Critical Functions**:
```python
def main():
    # App startup and configuration
    # Session state initialization
    # Model solving coordination
    # UI rendering

def solve_benchmark_unified():
    # Unified interface for local/API solving
    
def create_counterfactual_ui():
    # Dynamic UI for counterfactual configuration
```

### 2. Visualization Engine (`visualization.py`)

**Architecture**:
```python
class ModelVisualizationEngine:
    ├── VisualizationDataProcessor  # Data transformation
    ├── VisualizationUI            # UI components
    └── PlotlyVisualizer          # Chart generation
```

**Key Features**:
- Variable-specific UI generation
- Multi-dimensional data handling (1D, 2D, 3D, 4D)
- Percentage change calculations
- Interactive figure sizing

### 3. Model Pipeline (`model_pipeline.py`)

**Purpose**: Model solving orchestration and caching
**Key Classes**:
```python
class ModelPipeline:
    ├── BenchmarkModelSolver      # Baseline model solving
    ├── CounterfactualModelSolver # Scenario analysis
    └── ModelResultStorage       # Solution caching
```

**Caching Layers**:
- `@st.cache_resource`: Pipeline instance caching
- `@st.cache_data`: Function-level result caching
- Session state: User session persistence

### 4. API Integration (`api_client.py`, `api_server.py`)

**Purpose**: Remote model solving capability
**Benefits**:
- Scalable computation on dedicated servers
- Reduced local resource usage
- Consistent interface with local mode

## User Interface Design

### Navigation Flow

```
App Startup
    ↓
Mode Selection (Local/API)
    ↓
Model Type Selection
    ├── Baseline Model
    │   ├── Automatic Loading
    │   ├── Variable Selection
    │   ├── Visualization
    │   └── Downloads
    └── Counterfactual Model
        ├── Configuration UI
        ├── Scenario Solving
        ├── Results Comparison
        └── Downloads
```

### UI Components

#### 1. **Sidebar Controls**
- API mode toggle
- Cache management
- Server status indicators

#### 2. **Main Interface**
- Model type selection (radio buttons)
- Progress indicators
- Status messages
- Download sections

#### 3. **Visualization Controls**
- Variable selection (dropdown)
- Country/sector multiselect
- Figure size controls
- View mode selection

#### 4. **Download Interface**
- Excel export (all variables)
- CSV export (1D/2D variables)
- Percentage change downloads
- Timestamp-based filenames

## Data Flow

### Startup Sequence

1. **App Initialization**
   - Streamlit configuration
   - Module imports
   - API client setup

2. **Session State Setup**
   - Initialize empty solution containers
   - Set up cache keys
   - Configure UI state

3. **Model Loading** ⚠️ **PERFORMANCE BOTTLENECK**
   - Check session state for cached solution
   - If missing: solve baseline model (10-30s)
   - Store results in session state

4. **UI Rendering**
   - Render interface components
   - Initialize visualization engine
   - Set up user controls

### User Interaction Flow

```
User Action → State Update → Model Operation → Cache Update → UI Refresh
```

**Example: Variable Selection**
```
selectbox change → session state key → retrieve cached data → update visualization
```

**Example: Counterfactual Solving**
```
button click → spinner display → model solving → session state update → results display
```

## Performance Optimization

### Current Optimizations

#### 1. **Session State Caching**
```python
# Baseline model caching
if st.session_state['baseline_solution'] is None:
    # Solve only once per session
    baseline_sol, baseline_params = solve_benchmark_unified()
    st.session_state['baseline_solution'] = baseline_sol
    st.session_state['baseline_params'] = baseline_params
```

#### 2. **Streamlit Function Caching**
```python
@st.cache_resource
def get_model_pipeline() -> ModelPipeline:
    return ModelPipeline()

@st.cache_data
def solve_benchmark_cached() -> Tuple[ModelSol, ModelParams]:
    pipeline = get_model_pipeline()
    return pipeline.ensure_benchmark_solved()
```

#### 3. **Variable-Specific Keys**
```python
# Unique keys prevent cross-variable cache invalidation
key=f"country_multiselect_{variable_name}"
```

### Recommended Optimizations

#### 1. **Lazy Loading (Critical)**
**Problem**: Baseline model solved immediately on startup
**Solution**: Defer solving until user actually needs the data

```python
def get_baseline_solution_lazy():
    """Solve baseline model only when needed."""
    if st.session_state.get('baseline_solution') is None:
        with st.spinner("Loading baseline model..."):
            sol, params = solve_benchmark_unified()
            st.session_state['baseline_solution'] = sol
            st.session_state['baseline_params'] = params
    return st.session_state['baseline_solution'], st.session_state['baseline_params']
```

#### 2. **Progressive Loading**
**Concept**: Load metadata first, solve model on demand
```python
# Fast metadata loading (no model solving)
countries, sectors = get_metadata_cached()  # ~0.1s

# Model solving only when visualization requested
if st.button("Load Baseline Model"):
    with st.spinner("Solving baseline model..."):
        baseline_sol, baseline_params = solve_benchmark_unified()
```

#### 3. **Visualization Optimization**
- Implement chart virtualization for large datasets
- Use Plotly's `streaming` mode for real-time updates
- Implement pagination for multi-chart displays

#### 4. **Memory Management**
- Implement LRU cache for model solutions
- Clear unused session state variables
- Use memory-mapped arrays for large datasets

## Caching Strategy

### Multi-Level Caching Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Caching Hierarchy                             │
├─────────────────────────────────────────────────────────────────────┤
│  Level 1: Streamlit Function Cache (@st.cache_data)                  │
│  ├── solve_benchmark_cached()                                        │
│  ├── get_metadata_cached()                                          │
│  └── solve_counterfactual_cached()                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Level 2: Streamlit Resource Cache (@st.cache_resource)              │
│  ├── get_model_pipeline()                                           │
│  └── ModelPipeline instance                                         │
├─────────────────────────────────────────────────────────────────────┤
│  Level 3: Session State Cache (st.session_state)                     │
│  ├── baseline_solution                                              │
│  ├── baseline_params                                                │
│  ├── cf_solution                                                    │
│  └── cf_params                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Level 4: Application Cache (ModelResultStorage)                     │
│  ├── _benchmark_solution                                            │
│  ├── _counterfactual_solutions                                      │
│  └── _counterfactual_params                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Cache Invalidation Strategy

**Manual Cache Clearing**:
```python
if st.sidebar.button("Clear All Caches"):
    st.cache_data.clear()
    st.cache_resource.clear()
    # Clear session state
    for key in ['baseline_solution', 'baseline_params', 'cf_solution']:
        if key in st.session_state:
            st.session_state[key] = None
```

**Automatic Cache Management**:
- Session state persists within user session
- Function caches persist across sessions
- Resource caches survive app restarts

## Deployment Considerations

### Local Development
- Fast iteration with hot reload
- Full debugging capabilities
- Direct file system access

### Streamlit Cloud Deployment
- Automatic dependency management
- Shared resource constraints
- Cold start optimization critical

### API-Based Deployment
- Scalable backend processing
- Reduced frontend resource usage
- Network latency considerations

### Performance Recommendations by Environment

| Environment | Startup Strategy | Caching Priority | Resource Limits |
|-------------|------------------|------------------|-----------------|
| Local Dev | Lazy loading | Session state | Memory: 8GB+ |
| Streamlit Cloud | Progressive loading | Function cache | Memory: 1GB |
| API Mode | Immediate loading | Server-side cache | Network: 100ms |

## Troubleshooting Guide

### Common Performance Issues

#### 1. **App Startup Lag**
**Symptoms**: Long "Running" indicator on first load
**Diagnosis**: Check if baseline model is being solved immediately
**Solution**: Implement lazy loading or progressive loading

#### 2. **Memory Exhaustion**
**Symptoms**: App crashes or becomes unresponsive
**Diagnosis**: Monitor session state size and cache usage
**Solution**: Implement cache size limits and cleanup

#### 3. **Visualization Lag**
**Symptoms**: Slow chart rendering or UI freezing
**Diagnosis**: Check dataset size and number of charts
**Solution**: Implement chart virtualization or pagination

#### 4. **Cache Invalidation Issues**
**Symptoms**: Stale data or unexpected re-solving
**Diagnosis**: Check cache key uniqueness and dependencies
**Solution**: Use more specific cache keys and proper invalidation

### Debugging Tools

```python
# Memory usage monitoring
import psutil
st.sidebar.write(f"Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

# Cache state inspection
st.sidebar.write("Cache Info:")
st.sidebar.write(f"Session state keys: {list(st.session_state.keys())}")

# Performance timing
import time
start_time = time.time()
# ... operation ...
st.write(f"Operation took: {time.time() - start_time:.2f}s")
```

## Future Enhancements

### Planned Optimizations

1. **Async Model Solving**
   - Background model solving
   - Progress streaming
   - Cancellation support

2. **Advanced Caching**
   - Redis integration for multi-user deployments
   - Disk-based cache persistence
   - Intelligent cache warming

3. **UI Improvements**
   - Chart virtualization
   - Infinite scroll for large datasets
   - Real-time progress indicators

4. **Performance Monitoring**
   - Built-in performance metrics
   - User experience tracking
   - Automated optimization suggestions

### Scalability Considerations

- **Multi-user Support**: Implement user-specific caching
- **Resource Management**: Dynamic resource allocation
- **Load Balancing**: Distribute computation across servers
- **Monitoring**: Real-time performance tracking

---

*Documentation Version: 1.0*  
*Last Updated: January 2025*  
*Compatible with: Streamlit 1.28+, Python 3.8+* 