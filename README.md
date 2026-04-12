<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=🔍%20Hallucination%20Hunter&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI%20doesn't%20always%20tell%20the%20truth.%20Now%20you%20can%20prove%20it.&descAlignY=58&descSize=16&descColor=ccccff" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/hrshjha/hallucination-hunter?style=for-the-badge&logo=github&color=f59e0b&labelColor=1a1a2e)](https://github.com/hrshjha/hallucination-hunter)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-22c55e?style=for-the-badge&logo=git&logoColor=white)](CONTRIBUTING.md)

<br/>

<table>
<tr>
<td align="center">🎯<br/><b>Claim-Level</b><br/>Verification</td>
<td align="center">⚡<br/><b>REST API</b><br/>Ready</td>
<td align="center">🧠<br/><b>NLI-Powered</b><br/>Reasoning</td>
<td align="center">📊<br/><b>Explainable</b><br/>Output</td>
<td align="center">🔌<br/><b>RAG Pipeline</b><br/>Compatible</td>
</tr>
</table>

<br/>

> *"Don't trust. Verify. Automatically."*

<br/>

**Made with ❤️ by [Harsh Kumar](https://github.com/hrshjha)**

<br/>

<a href="https://twitter.com/m_eharsh">
  <img src="https://img.shields.io/badge/Follow-%40m__eharsh-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"/>
</a>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

</div>

---

## 📌 The Problem

AI systems generate confident but incorrect outputs. No built-in mechanism exists to verify whether responses are grounded in source data.

---

## 💡 The Solution

Hallucination Hunter:
- Splits responses into claims  
- Verifies using NLI + semantic similarity  
- Outputs FAITHFUL / HALLUCINATED  

---

## 🚀 Features

- 🔎 Claim-level verification  
- ⚖️ Contradiction detection  
- 📊 Confidence scoring  
- ⚡ FastAPI backend  
- 🧠 Explainable output  

---

## ⚙️ Installation

```bash
git clone https://github.com/hrshjha/hallucination-hunter.git
cd hallucination-hunter

pip install -r requirements.txt
python -m spacy download en_core_web_sm

uvicorn app.main:app --host 0.0.0.0 --port 8000 
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": "The Eiffel Tower is in Paris.",
    "response": "The Eiffel Tower is in Berlin."
  }'
  📜 License

MIT License © 2026 Harsh Kumar

🌟 Support

⭐ Star the repo
🍴 Fork it
📢 Share it

<div align="center">

Made with ❤️ by Harsh Kumar

</div> ```
