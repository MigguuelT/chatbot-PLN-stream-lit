# 🤖 Dashboard NLP & Chatbot com Streamlit

Um aplicativo completo em Python que integra um Chatbot interativo com um Dashboard de Análise de Dados na mesma interface. O projeto realiza análise de sentimentos em tempo real utilizando modelos Deep Learning, processa os textos com técnicas de NLP e disponibiliza um painel gerencial para exploração dos dados gerados.

## 🚀 Principais Funcionalidades

* **Chatbot Interativo:** Interface de mensagens construída nativamente com os componentes de chat do Streamlit.
* **Inteligência Artificial (BERT):** Classificação de sentimentos (1 a 5 estrelas) em tempo real utilizando o modelo `nlptown/bert-base-multilingual-uncased-sentiment` da Hugging Face.
* **Pipeline de NLP (spaCy):** Pré-processamento de texto estruturado (Lematização, remoção de stopwords e pontuação) focado na extração de termos relevantes.
* **Engenharia de Logs:** Armazenamento contínuo das interações e métricas do modelo em um arquivo `.log` padronizado.
* **Dashboard Gerencial Integrado:** Painel interativo construído com Pandas para visualizar o volume de mensagens, a confiança média da IA e a distribuição de sentimentos.
* **Filtros Dinâmicos:** Capacidade de isolar feedbacks negativos diretamente no painel para facilitar a tomada de decisão.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Front-end & Dashboard:** [Streamlit](https://streamlit.io/)
* **Deep Learning:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) / PyTorch
* **Processamento de Linguagem Natural:** [spaCy](https://spacy.io/) (`pt_core_news_sm`)
* **Análise de Dados:** Pandas

## ⚙️ Como Executar o Projeto

**1. Clone o repositório:**
```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio