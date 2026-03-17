# 🤖 Dashboard NLP & Chatbot Híbrido (BERT + LLM)

Um aplicativo completo em Python que integra um Chatbot de atendimento com um Dashboard Analítico na mesma interface. Este projeto demonstra uma arquitetura de **IA Híbrida**, utilizando um modelo local (Deep Learning) para classificação matemática de sentimentos e uma API externa (LLM) para geração de respostas empáticas e contextualizadas em tempo real.

## 🚀 Principais Funcionalidades

* **Arquitetura Híbrida de IA:** Separação inteligente de responsabilidades entre modelos preditivos e generativos.
* **Classificação Local (BERT):** Utiliza o modelo `nlptown/bert-base-multilingual-uncased-sentiment` (via Hugging Face e PyTorch) para ler a mensagem do utilizador e classificar a frustração/satisfação de 1 a 5 estrelas.
* **Geração de Linguagem Natural (Gemini):** Integração com o Google Gemini (1.5 Flash) via Prompt Engineering. O LLM recebe a classificação do BERT em *background* e gera uma resposta perfeitamente adequada ao estado emocional do cliente.
* **Dashboard Gerencial Integrado:** Painel analítico construído com Pandas e componentes nativos do Streamlit para visualizar o volume de mensagens, métricas de confiança da IA e a trilha de auditoria completa (Mensagem -> Score -> Resposta do Bot).
* **Exportação de Dados:** Capacidade de descarregar os logs de atendimento diretamente em formato `.csv` para análises futuras em Data Science.

## 🛠️ Stack Tecnológica

* **Linguagem:** Python 3.10+
* **Front-end & Dashboard:** [Streamlit](https://streamlit.io/)
* **Machine Learning / NLP:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) & PyTorch
* **IA Generativa / LLM:** Google Generative AI (Gemini API)
* **Manipulação de Dados:** Pandas

## ⚙️ Como Executar o Projeto Localmente

**1. Clone o repositório:**
```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio
```

**2. Crie e ative um ambiente virtual (Recomendado: Conda):**

```bash
conda create --name ai_env python=3.10 -y
conda activate ai_env
```
**3. Instale as dependências:**

```bash
pip install -r requirements.txt
```
**4. Configure a API Key:**

Para que o chatbot funcione corretamente e consiga gerar as respostas, é necessário fornecer uma chave de API do Google AI Studio:

   - Na raiz do projeto, crie um diretório oculto chamado .streamlit.

   - Dentro deste diretório, crie um ficheiro chamado secrets.toml.

   - Adicione a sua chave no ficheiro com o seguinte formato:

```Ini, TOML
GEMINI_API_KEY = "sua_chave_de_api_aqui"
```
**5. Inicie a aplicação Streamlit:**

```bash
streamlit run chatbot_nlp_gemini.py
```

(Na primeira execução, o sistema fará o download do modelo BERT automaticamente, o que pode levar alguns minutos).