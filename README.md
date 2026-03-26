# Support Agent

Support Agent is an LLM-based customer support application project.

It is designed to build a domain-adapted intelligent assistant through a complete pipeline including **LoRA fine-tuning**, **RAG-based knowledge augmentation**, **multi-turn dialogue management**, **FastAPI deployment**, and **frontend interaction**.

The goal of this project is to turn a fine-tuned large language model into a usable customer support system for domain-specific scenarios, rather than simply calling a general-purpose LLM API.

---

## Overview

This project develops an intelligent customer support prototype for domain-specific tasks.

Compared with directly using a general LLM, this system combines model adaptation and external knowledge retrieval to improve response quality, factual accuracy, and controllability.

The project focuses on building a complete end-to-end pipeline, including model fine-tuning, retrieval enhancement, backend deployment, and frontend interaction.

---

## Features

- Domain adaptation of a large language model with **LoRA fine-tuning**
- **RAG-based retrieval** to enhance factual and scenario-specific responses
- Support for **multi-turn conversations**
- **FastAPI-based backend service** for inference
- Simple **frontend interface** for interactive testing
- Modular and extensible project structure

---

## Pipeline

The overall workflow is:

Domain data preparation
→ LoRA fine-tuning
→ Fine-tuned model loading
→ RAG integration
→ Context construction
→ FastAPI service
→ Frontend interaction

At runtime, the frontend sends user messages to the backend service.  
The backend organizes dialogue context, retrieves relevant knowledge, formats the input prompt, calls the model for generation, and returns the final response to the frontend.

---

## Tech Stack

- Python
- PyTorch
- Transformers
- PEFT / LoRA
- FastAPI
- FAISS
- HTML / CSS / JavaScript

---

## Project Structure

support_agent/
├── backend/              
│   ├── api/              
│   ├── inference/        
│   └── app.py            
├── configs/              
├── frontend/             
├── rag/                  
├── vectorstore/          
└── README.md

---

## Running the Project

### Start Backend

uvicorn backend.app:app --host 127.0.0.1 --port 8000

If you want to allow access from another machine:

uvicorn backend.app:app --host 0.0.0.0 --port 8000

### Start Frontend

Open the frontend page in a browser for interactive testing.

The frontend is responsible for sending user messages to the backend and displaying the generated responses.

---

## Notes

- The vector database can be rebuilt if needed
- Configuration files can be adjusted for different domains or models
- This project focuses on building a usable customer support pipeline rather than optimizing only a single module

---

## Future Work

- Add memory mechanisms
- Support tool calling
- Integrate more advanced agent capabilities
- Improve frontend interaction and deployment
- Extend to more complex real-world support scenarios
