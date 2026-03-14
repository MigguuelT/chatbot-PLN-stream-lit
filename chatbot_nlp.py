import streamlit as st
import spacy
import logging
import pandas as pd
import os
from transformers import pipeline

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS E LOGGING
# ==========================================
st.set_page_config(page_title="Chatbot NLP", page_icon="🤖", layout="centered")

logging.basicConfig(
    filename='historico_nlp_streamlit.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)

# ==========================================
# 2. CARREGAMENTO DE MODELOS (CACHE)
# ==========================================
@st.cache_resource
def carregar_modelos():
    try:
        nlp_modelo = spacy.load("pt_core_news_sm")
    except OSError:
        st.error("Erro: Modelo spaCy não encontrado. Rode no terminal: python -m spacy download pt_core_news_sm")
        nlp_modelo = None
        
    classificador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return nlp_modelo, classificador

nlp, classificador_sentimento = carregar_modelos()

# ==========================================
# 3. FUNÇÕES DE PROCESSAMENTO E LÓGICA
# ==========================================
def pre_processar_texto(texto):
    if not nlp:
        return texto.lower()
    doc = nlp(texto.lower().strip())
    tokens_limpos = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens_limpos)

def gerar_resposta_bot(label_estrelas):
    nota = int(label_estrelas.split()[0])
    if nota >= 4:
        return "Fico muito feliz em saber disso! Há mais alguma coisa que eu possa fazer por você?"
    elif nota <= 2:
        return "Sinto muito que sua experiência não tenha sido a melhor. Vou registrar seu feedback para melhorarmos."
    else:
        return "Entendi perfeitamente. Gostaria de adicionar mais algum detalhe?"

# ==========================================
# 4. INTERFACE DO CHATBOT
# ==========================================
st.title("🤖 Chatbot NLP Avançado")
st.markdown("Análise de sentimentos em tempo real com armazenamento de logs.")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "logs_pendentes" not in st.session_state:
    st.session_state.logs_pendentes = []

# Renderiza histórico
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua mensagem..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Análise
    resultado = classificador_sentimento(prompt)[0]
    label = resultado['label'] 
    score = resultado['score']
    
    # Resposta
    resposta_bot = gerar_resposta_bot(label)
    st.session_state.mensagens.append({"role": "assistant", "content": resposta_bot})
    with st.chat_message("assistant"):
        st.markdown(resposta_bot)
        
    # Prepara log
    st.session_state.logs_pendentes.append({
        "original": prompt,
        "label": label,
        "score": score
    })

# Botão para salvar logs
st.write("")
if st.button("💾 Encerrar e Salvar Log", type="primary"):
    if not st.session_state.logs_pendentes:
        st.warning("Nenhuma conversa para salvar.")
    else:
        with st.spinner("Processando textos e salvando logs..."):
            for item in st.session_state.logs_pendentes:
                texto_limpo = pre_processar_texto(item["original"])
                linha_log = f"ORIGINAL: '{item['original']}' | LIMPO: '{texto_limpo}' | SENT: {item['label']} | CONF: {item['score']:.4f}"
                logging.info(linha_log)
            st.session_state.logs_pendentes.clear()
            st.success("Conversa encerrada! Dados salvos com sucesso.")

st.divider()

# ==========================================
# 5. DASHBOARD DE ANÁLISE DE DADOS
# ==========================================
with st.expander("📊 Abrir Dashboard de Análise de Sentimentos"):
    def carregar_dados_log(caminho_log):
        if not os.path.exists(caminho_log):
            return pd.DataFrame()
            
        dados_processados = []
        with open(caminho_log, 'r', encoding='utf-8') as arquivo:
            for linha in arquivo:
                try:
                    partes = linha.split(" | ")
                    data_hora = partes[0].strip()
                    
                    sentimento_raw = partes[4]
                    nota = int(sentimento_raw.replace("SENT: ", "").replace(" stars", "").replace(" star", "").strip())
                    
                    confianca_raw = partes[5]
                    score = float(confianca_raw.replace("CONF: ", "").strip())
                    
                    texto_original = partes[2].replace("ORIGINAL: ", "").strip("' ")
                    texto_limpo = partes[3].replace("LIMPO: ", "").strip("' ")
                    
                    dados_processados.append({
                        "Data/Hora": pd.to_datetime(data_hora),
                        "Nota (Estrelas)": nota,
                        "Confiança da IA": score,
                        "Texto Original": texto_original,
                        "Texto Processado (spaCy)": texto_limpo
                    })
                except Exception:
                    continue 
                    
        return pd.DataFrame(dados_processados)

    df_logs = carregar_dados_log('historico_nlp_streamlit.log')

    if df_logs.empty:
        st.info("O painel está vazio. Interaja com o chat acima e salve os logs para gerar análises.")
    else:
        # --- Lógica do Filtro ---
        st.subheader("Filtros de Análise")
        mostrar_apenas_negativos = st.checkbox("🚨 Mostrar apenas avaliações negativas (1 e 2 estrelas)")
        
        # Aplica o filtro no DataFrame se a caixa estiver marcada
        if mostrar_apenas_negativos:
            df_logs = df_logs[df_logs["Nota (Estrelas)"] <= 2]
            st.warning("Visualizando apenas mensagens classificadas como negativas.")

        if df_logs.empty:
            st.success("Nenhum sentimento negativo encontrado nos registros filtrados!")
        else:
            # --- KPIs ---
            col1, col2 = st.columns(2)
            total_interacoes = len(df_logs)
            confianca_media = df_logs["Confiança da IA"].mean() * 100
            
            col1.metric("Total de Registros (Filtrados)", total_interacoes)
            col2.metric("Confiança Média da IA", f"{confianca_media:.1f}%")
            
            # --- Gráficos ---
            st.subheader("Distribuição das Avaliações")
            contagem_estrelas = df_logs["Nota (Estrelas)"].value_counts().sort_index()
            # Define a cor vermelha para alertas negativos ou azul para a visão geral
            cor_grafico = "#d62728" if mostrar_apenas_negativos else "#1f77b4"
            st.bar_chart(contagem_estrelas, color=cor_grafico)
            
            # --- Tabela de Dados ---
            st.subheader("Visualização Detalhada dos Textos")
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
            
        if st.button("🔄 Atualizar Dashboard"):
            st.rerun()