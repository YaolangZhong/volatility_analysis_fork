# Deployment Instructions

## Main Entry Point

**Use `Furusawa2023/API/app.py` as the main entry point for deployment.**

### Why not `streamlit.py`?

The `streamlit.py` file in the root directory is just a launcher script that:
- Changes directories 
- Runs subprocess calls
- Has path resolution issues in deployment environments

### Deployment Configuration

For different platforms:

**Streamlit Cloud:**
```
Main file path: Furusawa2023/API/app.py
Python version: 3.8+
Requirements file: Furusawa2023/requirements.txt
```

**Heroku:**
```
web: streamlit run Furusawa2023/API/app.py --server.port $PORT
```

**Docker:**
```dockerfile
CMD ["streamlit", "run", "Furusawa2023/API/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Local Development:**
```bash
# From project root
cd Furusawa2023/API
streamlit run app.py

# Or from project root (recommended)
streamlit run Furusawa2023/API/app.py
```

### File Structure
```
Furusawa2023/
├── API/
│   ├── app.py              ← MAIN ENTRY POINT
│   ├── model_pipeline.py
│   ├── visualization.py
│   └── ...
├── streamlit.py            ← Launcher (for local dev only)
├── models.py
├── equations.py
└── ...
```

### Environment Variables

If using API mode, set:
```
API_URL=http://your-api-server:8000
``` 