# Multi-Agent Research Paper Reviewer

A modern, multi-agent academic peer-review simulator built with **FastAPI**, **CrewAI**, **Retrieval-Augmented Generation (RAG)**, and a beautiful **Vite** frontend.

This application automates the process of peer-reviewing scientific papers by simulating a panel of specialized academic reviewers, followed by a meta-reviewer (Judge Agent) that synthesizes their evaluations and produces a weighted final Accept/Reject verdict.

---

## 🚀 Key Features

- **Multi-Agent Consensus Panel**: Simulates peer reviewers with diverse personas:
  - **Methodology Reviewer**: Evaluates experimental design, statistical rigor, and reproducibility.
  - **Novelty Reviewer**: Assesses originality, contributions, and positioning against existing work.
  - **Clarity Reviewer**: Inspects structure, flow, clarity, and readability.
  - **Limitations Reviewer**: uncovers unstated limitations, ethical risks, and overclaims.
  - **Judge Agent (Program Chair)**: Synthesizes individual reports into a unified Accept/Reject verdict.
- **RAG-based Retrieval**: Uses Mistral AI Embeddings and an in-memory vector store to let agents perform similarity search queries on uploaded PDFs/CSVs.
- **FastAPI Backend**: Exposes endpoints for uploading manuscripts and streaming CrewAI logs in real-time using Server-Sent Events (SSE).
- **Vite Frontend**: A responsive dashboard containing:
  - **Submission Center**: File upload interface (PDF/CSV) with drag-and-drop.
  - **Live Review Console**: A terminal interface streaming real-time logs from the agents.
  - **Final Synthesis Report**: Displays overall score, individual scores, strengths, weaknesses, revisions, and the verdict.
  - **Mock Mode**: Standalone simulation for offline frontend testing.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher (for frontend)

### 1. Clone the Repository
```bash
git clone https://github.com/Sandeep-10/research-paper-review.git
cd research-paper-review
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
Mistral_Api_key=your_mistral_api_key
mistral_api_key_2=your_alternate_mistral_api_key
```

### 3. Backend Setup
Install the python dependencies and start the FastAPI server:
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
The backend server will run at `http://127.0.0.1:8000`.

### 4. Frontend Setup
Navigate to the `frontend` folder, install the packages, and run the dev server:
```bash
cd frontend
npm install
npm run dev
```
Open the URL shown in your terminal (typically `http://localhost:5173`) in your browser.

---

## 📊 Workflow Architecture

1. **Document Upload**: User uploads a PDF or CSV document.
2. **Chunking & Indexing**: The document is split into text chunks and indexed in an in-memory vector database using Mistral AI embeddings.
3. **Agent Reviews**: Specialized CrewAI agents query the vector store to review different aspects of the paper.
4. **Log Streaming**: Agent reasoning steps and milestones are streamed to the frontend via Server-Sent Events (SSE).
5. **Synthesis & Verdict**: The Judge Agent synthesizes the reviews, calculates the final score, and serves the results on the UI.
