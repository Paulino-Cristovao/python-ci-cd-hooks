# 🛠️ Python CI/CD & Hooks with GitHub Actions

This repository demonstrates **CI/CD pipelines with GitHub Actions** and **Python hooks** (pre-commit) to improve code quality, automation, and workflow efficiency.

## 🚀 Features
✅ CI/CD with GitHub Actions (Testing, Linting, Formatting)  
✅ Python hooks using `pre-commit`  
✅ Jupyter Notebooks for interactive examples  

---

## 📌 **1️⃣ GitHub Actions for Python CI/CD**
GitHub Actions automate testing, formatting, and linting.

### 🔹 Example Workflow: **CI with pytest**
Located in `.github/workflows/ci-pytest.yml`:
```yaml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run Tests
        run: pytest tests/
