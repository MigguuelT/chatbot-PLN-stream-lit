import streamlit as st
import logging
import pandas as pd
import os
from transformers import pipeline
import google.generativeai as genai

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS E LOGGING
# ==========================================
st.set_page_config(page_title="Chatbot NLP + Gemini", page_icon="🤖", layout="centered")

logging.basicConfig(
    filename='historico_nlp_streamlit.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)

# Configuração da API do Gemini usando os Secrets do Streamlit
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    modelo_gemini = genai.GenerativeModel('gemini-2.5-flash')
except KeyError:
    st.error("⚠️ Chave de API do Gemini não encontrada. Verifique seu arquivo .streamlit/secrets.toml")
    modelo_gemini = None

# ==========================================
# 2. CARREGAMENTO DO MODELO LOCAL (BERT)
# ==========================================
@st.cache_resource
def carregar_classificador():
    # Mantemos o BERT para garantir a consistência dos dados numéricos no Dashboard
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classificador_sentimento = carregar_classificador()

# ==========================================
# 3. LÓGICA DE GERAÇÃO (GEMINI)
# ==========================================
def gerar_resposta_gemini(prompt_usuario, nota_estrelas):
    if not modelo_gemini:
        return "Erro: Sistema de IA offline por falta de credenciais."
        
    # Prompt de Sistema (System Prompt) para guiar a postura do Gemini
    contexto = f"""
    Você é um assistente de atendimento ao cliente excepcional e empático.
    O usuário disse o seguinte: "{prompt_usuario}"
    Um sistema paralelo classificou o sentimento dessa mensagem como: {nota_estrelas} estrelas (de 1 a 5).
    
    Se a nota for 1 ou 2, seja muito acolhedor, peça desculpas pelo transtorno e ofereça solução.
    Se for 4 ou 5, seja entusiasmado e agradeça.
    Se for uma pergunta técnica, responda de forma clara e direta.
    
    Responda apenas com a fala do bot, sem aspas, de forma natural e concisa.
    """
    
    try:
        resposta = modelo_gemini.generate_content(contexto)
        return resposta.text.strip()
    except Exception as e:
        return f"Desculpe, tive um problema ao processar sua resposta. Erro: {e}"

# ==========================================
# 4. INTERFACE DO CHATBOT
# ==========================================
st.title("🤖 Chatbot Híbrido: BERT + Gemini")
st.markdown("Classificação local de sentimentos guiando uma IA Generativa.")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "logs_pendentes" not in st.session_state:
    st.session_state.logs_pendentes = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua mensagem..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Análise Matemática de Sentimento com BERT
    resultado = classificador_sentimento(prompt)[0]
    label = resultado['label'] 
    score = resultado['score']
    
    # Geração de Texto com Gemini Baseado no Sentimento
    resposta_bot = gerar_resposta_gemini(prompt, label)
    
    st.session_state.mensagens.append({"role": "assistant", "content": resposta_bot})
    with st.chat_message("assistant"):
        st.markdown(resposta_bot)
        
    # Prepara log com TODOS os dados
    st.session_state.logs_pendentes.append({
        "original": prompt,
        "label": label,
        "score": score,
        "bot_response": resposta_bot
    })

# Botão para salvar logs
st.write("")
if st.button("💾 Encerrar e Salvar Log", type="primary"):
    if not st.session_state.logs_pendentes:
        st.warning("Nenhuma conversa para salvar.")
    else:
        with st.spinner("Gravando interações no banco de dados..."):
            for item in st.session_state.logs_pendentes:
                # Novo formato de log incluindo o score e a resposta do bot
                linha_log = f"ORIGINAL: '{item['original']}' | SENT: {item['label']} | CONF: {item['score']:.4f} | BOT: '{item['bot_response']}'"
                logging.info(linha_log)
            st.session_state.logs_pendentes.clear()
            st.success("Conversa encerrada! Dados salvos com sucesso.")

st.divider()

# ==========================================
# 5. DASHBOARD DE ANÁLISE DE DADOS
# ==========================================
with st.expander("📊 Abrir Dashboard Gerencial de Atendimento"):
    def carregar_dados_log(caminho_log):
        if not os.path.exists(caminho_log):
            return pd.DataFrame()
            
        dados_processados = []
        with open(caminho_log, 'r', encoding='utf-8') as arquivo:
            for linha in arquivo:
                try:
                    # Formato: DATA | INFO | ORIGINAL: '...' | SENT: ... | CONF: ... | BOT: '...'
                    partes = linha.split(" | ")
                    data_hora = partes[0].strip()
                    texto_original = partes[2].replace("ORIGINAL: ", "").strip("' ")
                    nota = int(partes[3].replace("SENT: ", "").replace(" stars", "").replace(" star", "").strip())
                    score = float(partes[4].replace("CONF: ", "").strip())
                    texto_bot = partes[5].replace("BOT: ", "").strip("'\n ")
                    
                    dados_processados.append({
                        "Data/Hora": pd.to_datetime(data_hora).strftime("%Y-%m-%d %H:%M"),
                        "Mensagem do Cliente": texto_original,
                        "Nota (1-5)": nota,
                        "Score Confiança": f"{score:.2%}", # Formatado como porcentagem para ficar mais claro
                        "Resposta do Gemini": texto_bot
                    })
                except Exception:
                    continue 
                    
        return pd.DataFrame(dados_processados)

    df_logs = carregar_dados_log('historico_nlp_streamlit.log')

    if df_logs.empty:
        st.info("O painel está vazio. Interaja com o chat e salve os logs.")
    else:
        st.subheader("Filtros de Análise")
        mostrar_apenas_negativos = st.checkbox("🚨 Mostrar apenas avaliações negativas (1 e 2 estrelas)")
        
        if mostrar_apenas_negativos:
            df_logs = df_logs[df_logs["Nota (1-5)"] <= 2]

        if not df_logs.empty:
            col1, col2 = st.columns(2)
            col1.metric("Total de Registros (Filtrados)", len(df_logs))
            
            # Gráfico
            st.subheader("Distribuição das Avaliações")
            contagem_estrelas = df_logs["Nota (1-5)"].value_counts().sort_index()
            cor_grafico = "#d62728" if mostrar_apenas_negativos else "#1f77b4"
            st.bar_chart(contagem_estrelas, color=cor_grafico)
            
            # Tabela atualizada com a Resposta do Usuário, Resposta do Bot e o Score de Confiança
            st.subheader("Trilha de Auditoria e Scores")
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
            
            # Exportação
            csv = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Baixar Relatório em CSV", data=csv, file_name="relatorio_atendimento.csv", mime="text/csv", type="primary")
            
        if st.button("🔄 Atualizar Dashboard"):
            st.rerun()