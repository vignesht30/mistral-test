# Mistral AI: Advanced Implementation
### Tool Calling, RAG, and Beyond

This repository explores the capabilities of the Mistral AI API, focusing on building intelligent agents that can interact with external data and execute functional tools.

---

## 🚀 Features

* **Tool Calling (Function Calling):** Empowering the model to define and execute custom functions to interact with external APIs or local code.
* **Retrieval-Augmented Generation (RAG):** Enhancing model responses by injecting context from a custom vector database or document set.
* **Agentic Workflow:** Using Mistral as an orchestrator to decide when to use a tool vs. when to answer from internal knowledge.

---

## 🛠️ Tech Stack

* **LLM:** Mistral Large / Mistral 7B / Mixtral S]
* **Language:** Python 3.12+

---

## 📋 Prerequisites

1.  **Mistral API Key:** Get yours at [La Plateforme](https://console.mistral.ai/).
2.  **Environment Variables:** Create a `.env` file in the root directory:
    ```bash
    MISTRAL_API_KEY=your_api_key_here
    ```

---

## 📖 Project Structure



* `exp1/`: experiemnts- Tool Calling and RAG setup.


## 🔧 Getting Started

### 1. Installation
```bash
git clone [https://github.com/your-username/mistral-api-project.git](https://github.com/your-username/mistral-api-project.git)
cd mistral-api-project
pip install -r requirements.txt