# Network Visualization

This directory contains the network visualization components for the trade model.

## 🎯 **Main Visualization**

### **simple_3d_viz.html** ⭐ (Recommended)
- **Purpose**: Simple, reliable 3D network visualization
- **Features**: 
  - 37 country nodes with real expenditure data
  - Color-coded by continent
  - Interactive 3D navigation (drag to rotate, scroll to zoom)
  - Click nodes for country information
  - Real-time statistics panel
- **Usage**: Open directly in browser or use `open_3d_viz.py`

## 🚀 **How to Run**

### **Option 1: From Main Directory**
```bash
python run_network_viz.py
```

### **Option 2: Direct Access**
```bash
cd network_visualization
python open_3d_viz.py
```

### **Option 3: Manual Browser**
```bash
# Start HTTP server
python -m http.server 8000
# Then open: http://localhost:8000/network_visualization/simple_3d_viz.html
```

## 📁 **File Structure**

### **Essential Files (Keep)**
- `simple_3d_viz.html` - Main working 3D visualization
- `open_3d_viz.py` - Helper script to open visualization
- `README.md` - This documentation

### **Legacy Files (Can be removed)**
- `network_3d.html` - Complex 3D visualization (may have issues)
- `network_d3.html` - D3.js visualization
- `network_graph.py` - Network graph engine
- `network_app.py` - Streamlit app
- `network_data_generator.py` - Data generator
- `open_network_viz.py` - Legacy opener

## 🎮 **Controls**

### **Mouse Controls**
- **Drag**: Rotate the 3D view
- **Scroll**: Zoom in/out
- **Click**: Select node for information

### **Visual Elements**
- **Node Size**: Represents total expenditure
- **Colors**: Indicate continent groupings
  - 🟠 Orange: North America
  - 🔵 Blue: Europe  
  - 🔴 Red: Asia
  - 🟢 Green: South America
  - 🟣 Purple: Africa
  - 🟤 Brown: Oceania

## 📊 **Data**

The visualization shows 37 countries with their aggregate expenditure data:
- **Total Countries**: 37
- **Total Expenditure**: ~$220B
- **Data Source**: Trade model baseline scenario

## 🔧 **Troubleshooting**

### **If visualization doesn't load:**
1. Check browser console for JavaScript errors (F12)
2. Ensure JavaScript is enabled
3. Try refreshing the page
4. Use a modern browser (Chrome, Firefox, Safari)

### **If HTTP server needed:**
```bash
cd network_visualization
python -m http.server 8000
# Then open: http://localhost:8000/simple_3d_viz.html
```

## 🎯 **Recommendation**

Use `simple_3d_viz.html` as the primary visualization. It's:
- ✅ Reliable and stable
- ✅ Simple to understand
- ✅ Fast loading
- ✅ Cross-browser compatible
- ✅ Self-contained (no external dependencies) 