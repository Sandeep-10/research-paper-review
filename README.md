# Research Paper Reviewer

A multi-agent research paper reviewer built with **CrewAI** and **Retrieval-Augmented Generation (RAG)** using a **Gradio** web interface.

This application automates the process of peer-reviewing scientific papers by simulating a panel of five academic reviewer personas, followed by a meta-reviewer (Judge Agent) that synthesizes their evaluations and produces a weighted final accept/reject verdict.

## 🚀 Features

- **Multi-Agent Consensus Panel**: Simulates academic reviewers using diverse personas:
  - **Dr. Marcus Reid (Methodology Reviewer)**: Reviews experimental design, statistical rigor, and reproducibility.
  - **Dr. Aiko Tanaka (Novelty Reviewer)**: Evaluates research originality and positioning against existing literature.
  - **Dr. Elena Vasquez (Clarity Reviewer)**: Inspects scientific writing quality, layout, structure, and readability.
  - **Dr. Samuel Okonkwo (Limitations Reviewer)**: Identifies unacknowledged limits, overclaims, and ethical implications.
  - **Prof. Diana Chen (Judge Agent)**: Synthesizes individual reviewer reports into a final Accept/Reject verdict.
- **RAG-based Retrieval**: Uses Mistral AI Embeddings and an in-memory vector store to let agents perform similarity search queries on the uploaded PDF.
- **Modern Gradio Web UI**: Clean interface supporting both light and dark modes.

## 🛠️ Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sandeep-10/research-paper-review.git
   cd research-paper-review
   ```

2. **Install Dependencies**:
   ```bash
   pip install gradio crewai langchain-community langchain-text-splitters langchain-mistralai python-dotenv pypdf
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   Mistral_Api_key=your_mistral_api_key
   mistral_api_key_2=your_alternate_mistral_api_key
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```
