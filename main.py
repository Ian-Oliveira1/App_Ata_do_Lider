#Versão 1.4.4
#última alteração no código 03/08/2026
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import ast
import unicodedata

import altair as alt
import numpy as np

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="App Ata Lider", layout="wide")
st.title("📊 Monitor de Ocorrências")

pagina_menu = st.sidebar.radio("Navegação de Funcionalidades",
    [
        "📋 Ocorrências",
        "📊 Análise de Indicadores"
    ]
)

# ==============================
# STATE (NAVEGAÇÃO)
# ==============================

if "pagina" not in st.session_state:
    st.session_state.pagina = "lista"

if "idx_selecionado" not in st.session_state:
    st.session_state.idx_selecionado = None

if "pagina_cards" not in st.session_state:
    st.session_state.pagina_cards = 0

if "similares_confirmados" not in st.session_state:
    st.session_state.similares_confirmados = []

if "pagina_analise" not in st.session_state:
    st.session_state.pagina_analise = "lista"

if "grupo_selecionado" not in st.session_state:
    st.session_state.grupo_selecionado = None

if "grupo_confirmado" not in st.session_state:
    st.session_state.grupo_confirmado = None

if "titulo_grupo" not in st.session_state:
    st.session_state.titulo_grupo = ""

if "df_indicador_analise" not in st.session_state:
    st.session_state.df_indicador_analise = None

if "filtros_analise" not in st.session_state:
    st.session_state.filtros_analise = {}

if "filtros_grupo" not in st.session_state:
    st.session_state.filtros_grupo = {}

if "checkbox_grupo" not in st.session_state:
    st.session_state.checkbox_grupo = {}
# ==============================
# FUNÇÕES
# ==============================

PADROES_EQUIVALENCIA = {
    "yokohama": ["yok", "yoko", "yk", "yokohama", "yokohamas", "tokohama", "yokoghama", "ypokohama", "ykohama"],
    "upack": ["u pack", "u-pack", "upack", "umpack"],
    "flowpack": ["flow", "flowpack"],
    "automação": ["auto", "automacao"],
    "raio x": ["raio x", "raiox", "raio-x"],
    "panda": ["panda"],
    "balança": ["balan", "balanca", "bizerba"],
    "encaixotadora": ["encaixot"],
    "fabrima": ["fabrima"],
    "gima": ["gima"],
    "carimbo": ["carimbo"],
    "encartuchadeira": ["encartuch"],
    "rb80": ["rb80", "r80", "rb"],
    "trepko": ["trepko"],
    "sistema de visão": ["sistema visao", "sistema visão", "visao"],
    "geral": ["geral"],
    "seladora":["seladora", "selador"]
}

PADROES_INDICADORES = {
    "SEGURANÇA": [
        "segurança",
        "seguranca"
    ],

    "QUALIDADE": [
        "qualidade"
    ],

    "VOLUME": [
        "volume"
    ],

    "PARADA TOTAL": [
        "parada total"
    ],

    "PARADA POR FALHA": [
        "parada por falha",
        "falha"
    ],

    "PERDA DE NAKAMI": [
        "perda nakami",
        "perda de nakami"
    ],

    "PERDA DE EMBALAGEM": [
        "perda embalagem",
        "perda de embalagem"
    ],

    "GERAL": [
        "geral"
    ],

    "NENHUM": [
        "nenhum"
    ],

    "HORA HORA": [
        "hora hora"
    ]
}

def clean_columns(df):
    df.columns = [col.strip() for col in df.columns]
    return df


def parse_list(value):
    if pd.isna(value):
        return []

    value = str(value).strip().lower()

    # ======================
    # CASO JSON (Yokohama / E11)
    # ======================
    if value.startswith("[") and value.endswith("]"):
        try:
            lista = ast.literal_eval(value)
            if isinstance(lista, list):
                return [str(v).strip().lower() for v in lista]
        except:
            pass

    # ======================
    # PROTEGER EXPRESSÕES
    # ======================
    substituicoes = {
        "u pack": "upack",
        "u-pack": "upack",
        "flow pack": "flowpack",
        "raio x": "raio_x",
        "sistema de visao": "sistema_visao",
        "sistema de visão": "sistema_visao",
        "pan da": "panda",
        "fabrima arv": "fabrima",
    }

    for termo, substituto in substituicoes.items():
        value = value.replace(termo, substituto)

    # ======================
    # LIMPEZA PADRÃO
    # ======================
    value = re.sub(r"[,\-/]", " ", value)

    # remove palavras inúteis
    stop_remover = {"de", "da", "do", "e", "em"}

    tokens = value.split()

    tokens = [
        t.replace("_", " ")  # restaura espaços
        for t in tokens
        if t not in stop_remover
    ]

    return list(set(tokens))


def normalizar_itens_brutos(valor, linha=None):
    if pd.isna(valor):
        return []

    texto = str(valor).lower()

    # ======================
    # LIMPEZA BASE
    # ======================
    texto = texto.replace("\n", " ")
    texto = texto.replace("u-pack", "upack")
    texto = texto.replace("u pack", "upack")
    texto = re.sub(r"[,;/]", " e ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    # ======================
    # QUEBRAR EM PARTES
    # ======================
    partes = texto.split(" e ")

    resultados = []

    contexto_base = None

    for parte in partes:
        parte = parte.strip()

        # ======================
        # detectar número
        # ======================
        numeros = re.findall(r"\d+", parte)
        numero = numeros[0].lstrip("0") if numeros else None

        # ======================
        # REGRA: maquina X
        # ======================
        if "maquina" in parte and numero:

            # REGRA BASEADA NA LINHA
            if linha and linha.lower().strip() == "linha o":
                resultados.append(f"upack {numero}")
            else:
                resultados.append(f"yokohama {numero}")

            continue

        # ======================
        # detectar padrão base
        # ======================
        encontrado = None

        for padrao, variacoes in PADROES_EQUIVALENCIA.items():
            if any(v in parte for v in variacoes):
                encontrado = padrao
                contexto_base = padrao
                break

        # ======================
        # CASO: só número (ex: "2" depois de "yk 1 e 2")
        # ======================
        if not encontrado and numero and contexto_base == "yokohama":
            resultados.append(f"yokohama {numero}")
            continue

        if not encontrado:
            continue

        # ======================
        # adicionar com número se aplicável
        # ======================
        if encontrado in ["yokohama", "upack"] and numero:
            resultados.append(f"{encontrado} {numero}")
        else:
            resultados.append(encontrado)

    return list(dict.fromkeys(resultados))


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in ["", "n/a", "na", "**"]:
        return ""
    return value.lower()


def normalizar_indicadores(lista_indicadores):

    texto = " ".join(
        str(x).lower()
        for x in lista_indicadores
    )

    resultado = []

    # ======================
    # SEGURANÇA
    # ======================

    if "seguran" in texto:
        resultado.append("SEGURANÇA")

    # ======================
    # QUALIDADE
    # ======================

    if "qualidade" in texto:
        resultado.append("QUALIDADE")

    # ======================
    # VOLUME
    # ======================

    if "volume" in texto:
        resultado.append("VOLUME")

    # ======================
    # PARADA TOTAL
    # ======================

    if (
        "parada total" in texto
        or "total de parada" in texto
        or "total de paradas" in texto
        or "paradas totais" in texto
    ):
        resultado.append("PARADA TOTAL")

    # ======================
    # PARADA POR FALHA
    # ======================

    if (
        "parada por falha" in texto
        or "paradas por falha" in texto
        or "por falha" in texto
        or "falha" in texto
    ):
        resultado.append("PARADA POR FALHA")

    # ======================
    # PERDA DE NAKAMI
    # ======================

    if "nakami" in texto:
        resultado.append("PERDA DE NAKAMI")

    # ======================
    # PERDA DE EMBALAGEM
    # ======================

    if (
        "embalagem" in texto
        or "embalgem" in texto
        or "embalegem" in texto
        or "emabalagem" in texto
    ):
        resultado.append("PERDA DE EMBALAGEM")

    # ======================
    # HORA HORA
    # ======================

    if "hora hora" in texto or "hora a hora" in texto:
        resultado.append("HORA HORA")

    # ======================
    # GERAL
    # ======================

    if "geral" in texto:
        resultado.append("GERAL")

    # ======================
    # CASOS SEM IMPACTO
    # ======================

    if (
        "nenhum" in texto
        or "não" == texto.strip()
        or "sem impacto" in texto
        or "sem desvio" in texto
        or "n/a" in texto
    ):
        resultado.append("NENHUM")

    return list(dict.fromkeys(resultado))


stopwords_pt = [
    # Artigos
    "a", "o", "as", "os", "um", "uma", "uns", "umas",

    # Preposições
    "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas",
    "para", "pra", "por", "com",
    "ao", "aos", "à", "às",
    "até", "sobre", "entre", "sem",

    # Conjunções
    "e", "ou", "mas", "porém", "porque", "pois", "então",

    # Pronomes
    "ele", "ela", "eles", "elas",
    "se", "isso", "isto", "aquele", "aquela",
    "este", "esta", "esses", "essas",

    # Verbos muito comuns (ruído)
    "ser", "estar", "ter", "haver",
    "foi", "era", "está", "estava",
    "teve", "tinha", "sendo", "são",

    #TESTE
    "turno", "inicio", "início"

    # 🔄 Formas comuns no texto
    #"inicio", "início", "final", "fim",
    #"inicio de", "final de",

    # 🏭 Contexto industrial genérico (remove ruído)
    #"linha", "equipamento", "maquina", "máquina",
    #"produção", "processo",

    # 📄 termos comuns pouco informativos
    #"devido", "após", "antes", "durante",
    #"mesmo", "mesma", "outro", "outra",

    # valores vazios/populares
    #"n", "nao", "não", "sim", "-", "--",

    # 🔧 palavras muito frequentes nos registros
    #"realizado", "realizada",
    #"ocorreu", "ocorrência", "ocorrencias",
    #"ajuste", "ajustada", "ajustado",

    # ⚙️ conectores comuns em frase operacional
    #"para", "na", "no", "com", "sem", "em"
]
@st.cache_data
def calcular_modelo(textos):

    vectorizer = TfidfVectorizer(
        stop_words=stopwords_pt,
        ngram_range=(1, 2)
    )

    matriz = vectorizer.fit_transform(textos)

    return vectorizer, matriz

@st.cache_data
def carregar_dados(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return padronizar_df(df)

def padronizar_df(df):

    df = clean_columns(df)
    
    df["TURNO"] = df["TURNO"].apply(padronizar_turno) #add depois

    # 🔍 detectar coluna de máquina
    colunas_maquina = ["Maquina", "Máquina", "Equipamento", "Equipamentos"]

    coluna_encontrada = None

    for col in colunas_maquina:
        if col in df.columns:
            coluna_encontrada = col
            break

    if coluna_encontrada:
        df = df.rename(columns={coluna_encontrada: "Maquina"})
    else:
        df["Maquina"] = ""

    # 🔍 detectar indicador
    colunas_indicador = ["INDICADOR FORA", "Indicador", "Indicador Fora"]

    for col in colunas_indicador:
        if col in df.columns:
            df = df.rename(columns={col: "INDICADOR FORA"})
            break

    if "INDICADOR FORA" not in df.columns:
        df["INDICADOR FORA"] = ""

    # parse
    df["Maquina"] = df["Maquina"].apply(parse_list)
    df["INDICADOR FORA"] = df["INDICADOR FORA"].apply(parse_list)
    df["INDICADOR FORA"] = df["INDICADOR FORA"].apply(normalizar_indicadores)
    df["INDICADORES_PADRONIZADOS"] = df["INDICADOR FORA"].copy()


    # NORMALIZAÇÃO (função sendo testada)
    
    df["Maquina"] = df.apply(
        lambda row: list(set(
            m
            for item in row["Maquina"]
            for m in normalizar_itens_brutos(item, row.get("Linha", ""))
        )),
        axis=1
    )

    # texto
    # ========================
    # 🔍 DETECTAR COLUNA AÇÃO
    # ========================

    colunas_acao = ["Ação", "Acao", "AÇÃO", "Ação Executada no turno"]

    coluna_acao = None

    for col in colunas_acao:
        if col in df.columns:
            coluna_acao = col
            break

    if coluna_acao:
        df = df.rename(columns={coluna_acao: "Ação"})
    else:
        df["Ação"] = "-"

    # ======================
    # 🔍 DETECTAR CAUSA
    # ======================

    if "Causa" not in df.columns:
        df["Causa"] = "-"

    # ======================
    # 🔍 DETECTAR FATO
    # ======================

    if "Fato" not in df.columns:
        df["Fato"] = ""

    # ======================
    # 🧠 LIMPEZA
    # ======================

    # Mantém texto original para exibição
    df["Fato"] = df["Fato"].fillna("")

    # Coluna utilizada pelo motor de similaridade
    df["Fato_Processado"] = (
        df["Fato"]
        .apply(clean_text)
        .apply(remover_acentos)
        .apply(remover_maquinas_texto)
        .apply(padronizar_termos_texto)
    )

    df["Causa"] = df["Causa"].fillna("-")
    df["Ação"] = df["Ação"].fillna("-")

    # data
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")

    # tempo
    if "Tempo de impacto" not in df.columns:
        df["Tempo de impacto"] = None

    return df



def padronizar_turno(value):
    if pd.isna(value):
        return "Desconhecido"

    value = str(value).lower()

    if "1" in value:
        return "1° Turno"
    elif "2" in value:
        return "2° Turno"
    elif "3" in value:
        return "3° Turno"
    else:
        return value

def mostrar_paginacao(total_paginas, prefixo):

    col_a, col_b, col_c = st.columns([1, 2, 1])

    with col_a:
        if st.button("⬅ Anterior", key=f"{prefixo}_prev"):
            st.session_state.pagina_cards = max(
                0,
                st.session_state.pagina_cards - 1
            )
            st.rerun()

    with col_b:
        st.markdown(
            f"""
            <div style="text-align:center">
                <b>Página {st.session_state.pagina_cards + 1} de {total_paginas}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_c:
        if st.button("Próxima ➡", key=f"{prefixo}_next"):
            st.session_state.pagina_cards = min(
                total_paginas - 1,
                st.session_state.pagina_cards + 1
            )
            st.rerun()

def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def remover_maquinas_texto(texto):

    texto = str(texto)

    padroes = [

        # ----------------------
        # YOKOHAMA / YOK / YK
        # ----------------------

        # yk 1 e 2
        r'\byk\s*\d+\s*e\s*\d+\b',

        # yok 1 e 2
        r'\byok\s*\d+\s*e\s*\d+\b',

        # yokohama 1 e 2
        r'\byokohama\s*\d+\s*e\s*\d+\b',

        # yk 1,2,3
        r'\byk\s*\d+(?:\s*,\s*\d+)+\b',

        # yok 1,2,3
        r'\byok\s*\d+(?:\s*,\s*\d+)+\b',

        # yokohama 1,2,3
        r'\byokohamas?\s*\d+(?:\s*,\s*\d+)+\b',

        # upack 1,2
        r'\bu[\s\-]?pack\s*\d+(?:\s*,\s*\d+)+\b',

        # yokohama 1
        r'\byokohama\s*\d+\b',

        # yok 1
        r'\byok\s*\d+\b',

        # yk 1
        r'\byk\s*\d+\b',

        # yokohama1
        r'\byokohama\d+\b',

        # yok1
        r'\byok\d+\b',

        # yk1
        r'\byk\d+\b',

        # yokohama / yokohamas
        r'\byokohamas?\b',

        # yok
        r'\byok\b',

        # yk
        r'\byk\b',

        # ----------------------
        # FLOWPACK
        # ----------------------

        r'\bflowpack\b',
        r'\bflow\b',

        # ----------------------
        # U-PACK / UPACK
        # ----------------------

        # upack 1 e 2
        r'\bu[\s\-]?pack\s*\d+\s*e\s*\d+\b',

        # upack 1
        r'\bu[\s\-]?pack\s*\d+\b',

        # upack1
        r'\bu[\s\-]?pack\d+\b',

        # upack
        r'\bu[\s\-]?pack\b',
    ]

    for padrao in padroes:

        texto = re.sub(
            padrao,
            " ",
            texto,
            flags=re.IGNORECASE
        )
    
    texto = re.sub(r'\s*-\s*', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto

def padronizar_termos_texto(texto):

    substituicoes = {

        # raio x
        r'\braio[\s\-]?x\b': 'raio_x',

    }

    for padrao, substituto in substituicoes.items():

        texto = re.sub(
            padrao,
            substituto,
            texto,
            flags=re.IGNORECASE
        )

    return texto
# ==============================
# UPLOAD + PROCESSAMENTO
# ==============================

if "df" not in st.session_state:

    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.stop()

    df = carregar_dados(uploaded_file)

    st.session_state.df = df

    # ======================
    # LISTAS PARA FILTROS
    # ======================

    st.session_state.lista_linhas = [
        "Todas as Linhas"
    ] + sorted(df["Linha"].dropna().unique().tolist())

    st.session_state.lista_maquinas = sorted(
        set(
            m
            for lista in df["Maquina"]
            for m in lista
        )
    )

    st.session_state.lista_turnos = sorted(
        df["TURNO"].dropna().unique().tolist()
    )


else:
    df = st.session_state.df

    if df is None:
        st.error("Nenhum dado carregado. Volte para a página inicial.")
        st.stop()

# ==============================
# SIMILARIDADE
# ==============================

textos = df["Fato_Processado"].fillna("").tolist()
vectorizer, matriz = calcular_modelo(textos)

# ==============================
# FILTROS
# ==============================


if (pagina_menu == "📋 Ocorrências" and st.session_state.pagina == "lista"):



    if "df" in st.session_state:
        if st.button("🔄 Trocar arquivo"):
            del st.session_state.df
            
            for key in [
                "dias",
                "linha_sel",
                "maq_sel",
                "turno_sel",
                "pagina_cards",
                "lista_linhas",
                "lista_maquinas",
                "lista_turnos"
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state.pagina = "lista"
            st.session_state.pagina_cards = 0
            st.rerun()


    # ======================
    # 🔍 BUSCA
    # ======================
    query = st.text_input("🔍 Buscar ocorrência (por similaridade do Fato)")

    col1, col2, col3, col4 = st.columns(4)
    
    # calcular range dinâmico
    max_dias = (df["Data"].max() - df["Data"].min()).days
    max_dias = max(1, max_dias)

    with col1:
        # ======================
        # INICIALIZAÇÃO
        # ======================
        if "dias" not in st.session_state:
            st.session_state.dias = 5

        
        if st.session_state.dias > max_dias:
            st.session_state.dias = max_dias


        # ======================
        # SLIDER CONTROLADO
        # ======================
        dias = st.slider("Últimos dias", 1, max_dias, value=st.session_state.dias,  )

        # ======================
        # ATUALIZA MANUALMENTE
        # ======================
        st.session_state.dias = dias


    df_base = df.copy()  # SEM FILTRO (usado para opções)
    df_filtrado = df[df["Data"] >= (datetime.now() - timedelta(days=dias))].copy()

    with col2:
        linhas = st.session_state.lista_linhas
        
        # inicializa
        if "linha_sel" not in st.session_state:
            st.session_state.linha_sel = "Todas as Linhas"

        # remove valores inválidos
        #st.session_state.linha_sel = [
            #l for l in st.session_state.linha_sel if l in linhas
        #]
        if st.session_state.linha_sel not in linhas:
            st.session_state.linha_sel = "Todas as Linhas"

        # multiselect
        #linha_sel = st.multiselect(
        linha_sel = st.selectbox(
            "Linha",
            linhas,
            #default=st.session_state.linha_sel
            index=linhas.index(st.session_state.linha_sel)
        )

        # atualiza
        st.session_state.linha_sel = linha_sel


    if linha_sel != "Todas as Linhas":
        df_filtrado = df_filtrado[df_filtrado["Linha"] == linha_sel]

    with col3:
        maquinas = st.session_state.lista_maquinas

        # ======================
        # INICIALIZA
        # ======================
        if "maq_sel" not in st.session_state:
            st.session_state.maq_sel = []

        # ======================
        # CORRIGE VALORES INVÁLIDOS
        # ======================
        st.session_state.maq_sel = [
            m for m in st.session_state.maq_sel if m in maquinas
        ]

        # ======================
        # MULTISELECT
        # ======================
        maq_sel = st.multiselect(
            "Máquina",
            maquinas,
            default=st.session_state.maq_sel
        )

        # ======================
        # ATUALIZA STATE
        # ======================
        st.session_state.maq_sel = maq_sel


    if maq_sel:
        df_filtrado = df_filtrado[
            df_filtrado["Maquina"].apply(lambda x: bool(set(x) & set(maq_sel)))
        ]

    with col4:
        turnos = st.session_state.lista_turnos

        # inicializa
        if "turno_sel" not in st.session_state:
            st.session_state.turno_sel = []

        # remove valores inválidos 
        st.session_state.turno_sel = [
            t for t in st.session_state.turno_sel if t in turnos
        ]

        # multiselect
        turno_sel = st.multiselect(
            "Turno",
            turnos,
            default=st.session_state.turno_sel
        )

        # atualiza
        st.session_state.turno_sel = turno_sel

    if turno_sel:
        df_filtrado = df_filtrado[df_filtrado["TURNO"].isin(turno_sel)]

    # ======================
    # 🔍 APLICAR BUSCA
    # ======================

    if query and len(query) >= 3:

        # vetor da busca
        query_limpa = remover_acentos(clean_text(query))
        query_vec = vectorizer.transform([query_limpa])

        # similaridade com todos os registros
        scores = cosine_similarity(query_vec, matriz)[0]

        # adiciona score ao dataframe
        df_filtrado = df_filtrado.copy()
        df_filtrado["score"] = scores[df_filtrado.index]

        # boost para match exato
        df_filtrado["score"] += df_filtrado["Fato_Processado"].apply(
            lambda x: 0.2 if query_limpa in x else 0
        )

        # filtra por relevância mínima
        df_filtrado = df_filtrado[df_filtrado["score"] > 0.1]

        # ordena por score (mais relevante primeiro)
        df_filtrado = df_filtrado.sort_values(by="score", ascending=False)

    else:
        # comportamento normal
        df_filtrado = df_filtrado.sort_values(by="Data", ascending=False)


else:
    # na página detalhe, usa dataset completo ou mantém filtrado anterior
    df_filtrado = df.copy()

# ==============================
# 🟢 PAGINA 1 — LISTA
# ==============================

if (pagina_menu == "📋 Ocorrências" and st.session_state.pagina == "lista"):
    
    #botão limpar filtro
    if st.button("⏹ Resetar filtros"):
        st.session_state.dias = 5
        st.session_state.linha_sel = "Todas as Linhas"
        st.session_state.maq_sel = []
        st.session_state.turno_sel = []
        st.session_state.pagina_cards = 0
        st.rerun()


    # botão de reset
    #if "df" in st.session_state:
        #if st.button("🔄 Trocar arquivo"):
            #del st.session_state.df
            
            #for key in [
                #"dias",
                #"linha_sel",
                #"maq_sel",
                #"turno_sel",
                #"pagina_cards",
                #"lista_linhas",
                #"lista_maquinas",
                #"lista_turnos"
            #]:
                #if key in st.session_state:
                    #del st.session_state[key]

            #st.session_state.pagina = "lista"
            #st.session_state.pagina_cards = 0
            #st.rerun()


    #INICIO DA PAGINAÇÂO###################################################

    por_pagina = 25 #<== TROCAR NUMERO DE CARDS POR PÁGINA É AQUI

    total_registros = len(df_filtrado)

    total_paginas = max(
        1,
        (total_registros - 1) // por_pagina + 1
    )
    
    if st.session_state.pagina_cards >= total_paginas:
        st.session_state.pagina_cards = 0

    if st.session_state.pagina_cards >= total_paginas:
        st.session_state.pagina_cards = total_paginas - 1

    inicio = st.session_state.pagina_cards * por_pagina
    fim = inicio + por_pagina

    df_pagina = df_filtrado.iloc[inicio:fim]
    #FIM DA PAGINAÇÂO######################################################


    st.subheader("📋 Ocorrências")

    mostrar_paginacao(total_paginas, "top")

    for idx, row in df_pagina.iterrows():

        maquinas_texto = ", ".join(row["Maquina"])
        indicador_texto = ", ".join(row["INDICADOR FORA"]) or "-"

        with st.container():

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"### {indicador_texto}")
                st.write(f"📍 **{row['Linha']} | {maquinas_texto}**")
                st.write(f"📝 {row['Fato'][:120]}...")

            with col2:
                st.write(row["Data"].strftime('%d/%m/%Y'))
                st.write(row["TURNO"])
                
                # tempo de impacto
                tempo = row.get("Tempo de impacto", "-")
                st.write(f"⏱ {tempo} min")


            # BOTÃO DE ABRIR DETALHE
            with col3:
                if st.button("Ver", key=f"btn_{idx}"):
                    st.session_state.similares_confirmados = []
                    st.session_state.idx_selecionado = idx
                    st.session_state.pagina = "detalhe"
                    st.rerun()

            st.divider()

    mostrar_paginacao(total_paginas, "bottom")
# ==============================
# 🔵 PAGINA 2 — DETALHE
# ==============================

if (pagina_menu == "📋 Ocorrências" and st.session_state.pagina == "detalhe"):

    idx = st.session_state.idx_selecionado
    row = df.loc[idx]

    # 🔙 VOLTAR
    if st.button("⬅️ Voltar"):
        st.session_state.similares_confirmados = []
        st.session_state.pagina = "lista"
        st.rerun()

    st.markdown("## 📌 Detalhe da Ocorrência")

    maquinas_texto = ", ".join(row["Maquina"])
    indicador_texto = ", ".join(row["INDICADOR FORA"]) or "-"

    st.markdown(f"### {indicador_texto}")
    st.write(f"📍 **{row['Linha']} | {maquinas_texto}**")
    tempo = row.get("Tempo de impacto", "-")
    st.write(f"📅 {row['Data'].strftime('%d/%m/%Y')} | {row['TURNO']} | ⏱ {tempo} min")

    st.markdown("### 📝 Fato")
    st.write(row["Fato"])

    st.markdown("### 🔍 Causa")
    st.write(row["Causa"])

    st.markdown("### ⚙️ Ação")
    st.write(row["Ação"])

    # ==============================
    # 🔍 CALCULAR SIMILARES
    # ==============================

    similares = cosine_similarity(
        matriz[idx],
        matriz
    )[0]
    indices_similares = similares.argsort()[::-1]

    # remover o próprio índice
    indices_similares = [i for i in indices_similares if i != idx][:500]

    lista_similares = []

    for i in indices_similares:

        outra = df.iloc[i]

        # mesma linha
        if outra["Linha"] != row["Linha"]:
            continue

        # interseção de máquina
        # regra especial para GERAL
        if "geral" in row["Maquina"]:
            pass  # aceita
        else:
            if not any(m in outra["Maquina"] for m in row["Maquina"]):
                continue

        # SENSIBILIDADE
        if similares[i] < 0.20:
            continue

        lista_similares.append((i, similares[i]))

    # ==============================
    # 📊 MÉTRICAS
    # ==============================

    total_similares = len(lista_similares)
    if not st.session_state.similares_confirmados:
        st.session_state.similares_confirmados = lista_similares.copy()
    #if (
        #len(st.session_state.similares_confirmados) == 0
    #):
        #st.session_state.similares_confirmados = lista_similares.copy()

    st.markdown("---")
    st.markdown(f"## 📊 Total de similares encontrados: **{total_similares}**")

    # ==============================
    # 🔎 LISTA DE SIMILARES (VERSÃO AVANÇADA)
    # ==============================

    st.markdown("## 🔎 Ocorrências similares")
    st.caption("Score de similaridade (0 a 1) — clique para expandir")

    if total_similares == 0:
        st.write("Nenhuma ocorrência similar relevante")

    else:
        if st.button("✅ Aplicar seleção"):
                #st.success("Seleção aplicada!")
                #st.toast("✅ Seleção aplicada!")
            
            similares_confirmados = []

            for pos, (i, score) in enumerate(lista_similares):

                considerar = st.session_state.get(
                    f"similar_{idx}_{pos}",
                    True
                )

                if considerar:
                    similares_confirmados.append(
                        (i, score)
                    )

            st.session_state.similares_confirmados = similares_confirmados

            st.toast("✅ Seleção aplicada!")
        
        st.caption(
            f"Ocorrências similares consideradas: {len(st.session_state.similares_confirmados)} de {total_similares}"
        )
        for pos, (i, score) in enumerate(lista_similares):

            outra = df.iloc[i]
            score = round(score, 2)

            maquinas_out = ", ".join(outra["Maquina"]) if outra["Maquina"] else "-"

            preview = (
                outra["Fato"][:80] + "..."
                if len(outra["Fato"]) > 80
                else outra["Fato"]
            )

            tempo = outra.get("Tempo de impacto", "-")

            col_check, col_exp = st.columns([1, 12])

            with col_check:

                considerar = st.checkbox(
                    "",
                    value=True,
                    key=f"similar_{idx}_{pos}"
                )

            icone_status = "🟢" if considerar else "🔴"

            with col_exp:

                with st.expander(
                    f"{icone_status} 🔎 Similar ({score}) | {outra['Data'].strftime('%d/%m/%Y')} | ⏱ {tempo} min | {preview}"
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"📍 **{outra['Linha']} | {maquinas_out}**")

                    with col2:
                        st.write(f"{outra['TURNO']}")
                        st.write(f"⏱ {outra.get('Tempo de impacto','-')} min")

                    st.markdown("**📝 Fato:**")
                    st.write(outra["Fato"] if outra["Fato"] else "-")

                    st.markdown("**🔍 Causa:**")
                    st.write(outra["Causa"] if outra["Causa"] else "-")

                    st.markdown("**⚙️ Ação:**")
                    st.write(outra["Ação"] if outra["Ação"] else "-")




        # ==============================
        # 📈 GRÁFICO AVANÇADO (TIMELINE)
        # ==============================

        import altair as alt
        import numpy as np

        if len(st.session_state.similares_confirmados) > 0:

            datas = []
            tipos = []

            # SIMILARES CONFIRMADOS
            for i, score in st.session_state.similares_confirmados:
                outra = df.iloc[i]
                datas.append(outra["Data"])
                tipos.append("Similar")

            # OCORRÊNCIA ATUAL
            datas.append(row["Data"])
            tipos.append("Atual")

            df_temp = pd.DataFrame({
                "Data": pd.to_datetime(datas),
                "Tipo": tipos
            })

            # criar leve variação no eixo Y (margem visual)
            np.random.seed(42)
            df_temp["y"] = np.random.uniform(0.49, 0.51, len(df_temp))

            st.markdown("---")
            st.markdown("### 📈 Ocorrências ao longo do tempo")

            grafico = alt.Chart(df_temp).mark_circle(size=100).encode(
                #x=alt.X("Data:T", title="Data"),
                x=alt.X(
                    "Data:T",
                    title="Data",
                    axis=alt.Axis(format="%d/%m/%y")
                ),
                y=alt.Y("y:Q", title="", axis=None, scale=alt.Scale(domain=[0.45, 0.55])),
                color=alt.Color(
                    "Tipo:N",
                    scale=alt.Scale(
                        domain=["Similar", "Atual"],
                        range=["#4cc9f0", "#ff4d4f"]  # azul e vermelho
                    ),
                    legend=alt.Legend(title="Tipo")
                ),
                #tooltip=["Data", "Tipo"]
                tooltip=[
                    alt.Tooltip("Data:T", title="Data", format="%d/%m/%Y"),
                    alt.Tooltip("Tipo:N", title="Tipo")
                ]
            ).properties(
                height=180
            )

            st.altair_chart(grafico, use_container_width=True)


        # ==============================
        # Estatísticas
        # ==============================
        st.markdown("---")
        st.markdown(
            """
            <div>
                <h1 style="margin-bottom:0px;">Estatísticas</h1>
                <p style="margin-top:-5px; font-size:15px; color:#888;">
                    (incluindo ocorrência atual)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==============================
        # ⏱ TEMPO DE IMPACTO
        # ==============================

        tempos = []
        tempos_validos = []

        for i, score in st.session_state.similares_confirmados:
            outra = df.iloc[i]

            tempo = outra.get("Tempo de impacto", None)

            try:
                tempo = float(tempo)
                tempos.append(tempo)

                if tempo > 0:
                    tempos_validos.append(tempo)

            except:
                continue

        # ocorrência atual
        tempo_atual = row.get("Tempo de impacto", None)

        try:
            tempo_atual = float(tempo_atual)
            tempos.append(tempo_atual)

            if tempo_atual > 0:
                tempos_validos.append(tempo_atual)

        except:
            pass

        # CONTROLE
        zeros_removidos = len(tempos) - len(tempos_validos)

##
        if len(tempos) > 0:

            st.markdown("---")
            st.markdown("## ⏱ Impacto das ocorrências")
            if zeros_removidos > 0:
                usados = len(tempos_validos)
            #    
                texto_zero = "ocorrência" if zeros_removidos == 1 else "ocorrências"
                verbo = "foi" if zeros_removidos == 1 else "foram"
                sufixo = "" if zeros_removidos == 1 else "s"

                texto_usadas = "usada" if usados == 1 else "usadas"

                st.caption(
                    f"⚠️ {zeros_removidos} {texto_zero} com tempo 0 "
                    f"{verbo} desconsiderada{sufixo} ({usados} {texto_usadas} no cálculo)"
                )

            # converter para série
            if len(tempos_validos) > 0:
                serie_tempos = pd.Series(tempos_validos)
            else:
                st.write("Sem dados válidos de impacto")
                serie_tempos = None

            # estatísticas
            if serie_tempos is not None:
                media = round(serie_tempos.mean(), 1)
                desvio = round(serie_tempos.std(), 1)
                maximo = int(serie_tempos.max())
                minimo = int(serie_tempos.min())

                st.write(f"Tempo médio de impacto: **{media} min**")
                st.write(f"Desvio padrão: {desvio} min")
                st.write(f"Tempo máximo: {maximo} min")
                st.write(f"Tempo mínimo: {minimo} min")

        # ==============================
        # 🔵 ANÁLISE POR TURNO
        # ==============================

        turnos_data = []
        turnos_data = []

        # similares confirmados
        for i, score in st.session_state.similares_confirmados:
            outra = df.iloc[i]

            turno = outra.get("TURNO", "N/A")
            tempo = outra.get("Tempo de impacto", None)
            acao_diaria = outra.get("AÇÃO NA DIARIA", None)

            try:
                tempo = float(tempo)
            except:
                tempo = None

            turnos_data.append({
                "Turno": turno,
                "Tempo": tempo,
                "Acao": acao_diaria if pd.notna(acao_diaria) else "Vazio"
            })

        # ocorrência atual
        turno_atual = row.get("TURNO", "N/A")
        tempo_atual = row.get("Tempo de impacto", None)
        acao_atual = row.get("AÇÃO NA DIARIA", None)

        try:
            tempo_atual = float(tempo_atual)
        except:
            tempo_atual = None

        turnos_data.append({
            "Turno": turno_atual,
            "Tempo": tempo_atual,
            "Acao": acao_atual if pd.notna(acao_atual) else "Vazio"
        })

        if len(turnos_data) > 0:

            df_turnos = pd.DataFrame(turnos_data)

            # ======================
            # REMOVER ZEROS PARA MÉDIA
            # ======================

            df_turnos_validos = df_turnos[df_turnos["Tempo"] > 0]

            zeros_turno = len(df_turnos) - len(df_turnos_validos)

            st.markdown("---")
            st.markdown("## 🔵 Análise por turno")

            # ======================
            # 📊 QUANTIDADE POR TURNO
            # ======================

            st.markdown("### 📊 Número de ocorrências por turno")

            turnos_ordem = ["1° Turno", "2° Turno", "3° Turno"]

            ocorrencias_turno = df_turnos["Turno"].value_counts()

            for t in turnos_ordem:
                valor = ocorrencias_turno.get(t, 0)
                st.write(f"{t}: {valor}")


            # ======================
            # ⏱ MÉDIA POR TURNO
            # ======================

            st.markdown("### ⏱ Tempo médio de impacto por turno")

            if zeros_turno > 0:
                usados = len(df_turnos_validos)
                texto_zero = "ocorrência" if zeros_turno == 1 else "ocorrências"
                verbo = "foi" if zeros_turno == 1 else "foram"
                sufixo = "" if zeros_turno == 1 else "s"

                texto_usadas = "usada" if usados == 1 else "usadas"

                st.caption(
                    f"⚠️ {zeros_turno} {texto_zero} com tempo 0 "
                    f"{verbo} desconsiderada{sufixo} ({usados} {texto_usadas} no cálculo)"
                )

            media_turno = df_turnos_validos.groupby("Turno")["Tempo"].mean()

            for t in turnos_ordem:
                valor = media_turno.get(t, None)

                if pd.notna(valor):
                    valor = round(valor, 1)
                    st.write(f"{t}: {valor} min")
                else:
                    st.write(f"{t}: sem dados válidos")
########################################################################################
# ==============================
# 📊 ANÁLISE DE INDICADORES
# ==============================

if (pagina_menu == "📊 Análise de Indicadores" and st.session_state.pagina_analise == "lista"):
    
    st.info("Funcionalidade em construção")
    st.header("📊 Análise de Indicadores")

    if "df" in st.session_state:

        if st.button(
            "🔄 Trocar arquivo",
            key="trocar_arquivo_analise"
        ):

            del st.session_state.df

            for key in [
                "dias",
                "linha_sel",
                "maq_sel",
                "turno_sel",
                "pagina_cards",
                "lista_linhas",
                "lista_maquinas",
                "lista_turnos",

                "linha_analise",
                "maq_analise",
                "turno_analise",
                "periodo_analise",
                "indicador_sel_analise",
                "filtros_analise"
            ]:

                if key in st.session_state:
                    del st.session_state[key]

            st.session_state.pagina = "lista"
            st.session_state.pagina_cards = 0

            st.rerun()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        linhas_analise = [
            "Todas"
        ] + sorted(
            df["Linha"].dropna().unique().tolist()
        )

        linha_analise = st.selectbox(
            "Linha",
            linhas_analise,
            key="linha_analise"
        )
    with col2:

        maquinas_analise = sorted(
            set(
                maquina
                for lista in df["Maquina"]
                for maquina in lista
            )
        )

        maq_analise = st.multiselect(
            "Máquina",
            maquinas_analise,
            key="maq_analise"
        )
    
    with col3:

        turnos_analise = sorted(
            df["TURNO"].dropna().unique().tolist()
        )

        turno_analise = st.multiselect(
            "Turno",
            turnos_analise,
            key="turno_analise"
        )
    
    with col4:

        periodo_analise = st.selectbox(
            "Período",
            [
                "Todos os dias",
                "Últimos 30 dias",
                "Últimos 90 dias",
                "Últimos 180 dias",
                "Últimos 365 dias"
            ],
            key="periodo_analise"
        )
    
    df_analise = df.copy()

    if periodo_analise != "Todos os dias":

        mapa_dias = {
            "Últimos 30 dias": 30,
            "Últimos 90 dias": 90,
            "Últimos 180 dias": 180,
            "Últimos 365 dias": 365
        }

        dias = mapa_dias[periodo_analise]

        df_analise = df_analise[
            df_analise["Data"] >= (
                datetime.now() - timedelta(days=dias)
            )
        ]

    if linha_analise != "Todas":
        df_analise = df_analise[
            df_analise["Linha"] == linha_analise
        ]

    if maq_analise:

        df_analise = df_analise[
            df_analise["Maquina"].apply(
                lambda x: bool(
                    set(x) & set(maq_analise)
                )
            )
        ]
    
    if turno_analise:

        df_analise = df_analise[
            df_analise["TURNO"].isin(turno_analise)
        ]

    indicadores_disponiveis = sorted(
        set(
            indicador
            for lista in df_analise["INDICADORES_PADRONIZADOS"]
            for indicador in lista
        )
    )

    if "indicador_sel_analise" not in st.session_state:
        st.session_state.indicador_sel_analise = None

    if (
        st.session_state.indicador_sel_analise
        not in indicadores_disponiveis
    ):

       if indicadores_disponiveis:
            st.session_state.indicador_sel_analise = (indicadores_disponiveis[0])

    indicador_sel = st.selectbox(
        "Indicador",
        indicadores_disponiveis,
        index=indicadores_disponiveis.index(
            st.session_state.indicador_sel_analise
        ),
        key="indicador_sel_analise"
    )

    st.session_state.filtros_analise = {
        "linha": linha_analise,
        "maq": maq_analise.copy(),
        "turno": turno_analise.copy(),
        "periodo": periodo_analise,
        "indicador": indicador_sel
    }
    
    df_indicador = df_analise[
        df_analise["INDICADORES_PADRONIZADOS"].apply(
            lambda x: indicador_sel in x
        )
    ]

    #st.session_state.df_indicador_analise = (df_indicador.copy())

    textos_indicador = (
        df_indicador["Fato_Processado"]
        .fillna("")
        .tolist()
    )

    if len(textos_indicador) > 1:

        vet_indicador = vectorizer.transform(
            textos_indicador
        )

        matriz_sim = cosine_similarity(
            vet_indicador
        )

    #st.metric(
        #"Total de ocorrências",
        #len(df_indicador)
    #)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
                "Total de ocorrências",
                len(df_indicador)
            )
        #primeira_data = df_indicador["Data"].min()
        #st.metric(
            #"Primeira ocorrência",
            #primeira_data.strftime("%d/%m/%Y")
        #)
    with col2:
        total_linhas = df_indicador["Linha"].nunique()
        
        st.metric(
            "Linhas afetadas",
            total_linhas
        )
        #ultima_data = df_indicador["Data"].max()
        #st.metric(
            #"Última ocorrência",
            #ultima_data.strftime("%d/%m/%Y")
        #)
    with col3:
        maquinas_unicas = set()

        for lista in df_indicador["Maquina"]:
            maquinas_unicas.update(lista)

        st.metric(
            "Máquinas afetadas",
            len(maquinas_unicas)
        )
        #total_linhas = df_indicador["Linha"].nunique()
        #st.metric(
            #"Linhas afetadas",
            #total_linhas
        #)
    with col4:

        #maquinas_unicas = set()
        #for lista in df_indicador["Maquina"]:
            #maquinas_unicas.update(lista)

        #st.metric(
            #"Máquinas afetadas",
            #len(maquinas_unicas)
        #)
        st.empty()
    st.markdown("---")
    st.subheader("📊 Distribuição das ocorrências")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🏭 Top Linhas")

        linhas_contagem = (
            df_indicador["Linha"]
            .value_counts()
            .head(10)
        )

        st.dataframe(
            linhas_contagem.rename("Ocorrências")
        )

    with col2:

        st.markdown("### ⚙️ Top Máquinas")
        maquinas_explodidas = (
            df_indicador[["Maquina"]]
            .explode("Maquina")
        )

        maquinas_contagem = (
            maquinas_explodidas["Maquina"]
            .value_counts()
            .head(10)
        )

        st.dataframe(
            maquinas_contagem.rename("Ocorrências")
        )

    
    #st.markdown("---")
    #st.subheader("📝 Ocorrências consideradas")

    #df_exibicao = df_indicador[
        #[
            #"Data",
            #"Linha",
            #"Fato"
        #]
    #].copy()

    #df_exibicao["Máquinas"] = df_indicador["Maquina"].apply(
        #lambda x: ", ".join(x)
    #)

    #df_exibicao = df_exibicao[
        #[
            #"Data",
            #"Linha",
            #"Máquinas",
            #"Fato"
        #]
    #]

    #st.dataframe(
        #df_exibicao.sort_values(
            #"Data",
            #ascending=False
        #),
        #use_container_width=True
    #)

    #st.markdown("---")
    #st.subheader("🔁 Fatos mais recorrentes")

    #fatos_validos = (
        #df_indicador["Fato"]
        #.dropna()
        #.astype(str)
        #.str.strip()
    #)

    #fatos_validos = fatos_validos[
        #fatos_validos != ""
    #]

    #top_fatos = (
        #fatos_validos
        #.value_counts()
        #.head(20)
    #)

    #st.dataframe(
        #top_fatos.rename("Ocorrências"),
        #use_container_width=True
    #)

    st.markdown("---")
    st.subheader("🧠 Agrupamentos Automáticos")

    if len(df_indicador) <= 1:

        st.info(
            "Poucas ocorrências para agrupamento."
        )
    else:
        #Depois apagar essa parte (se eu realmente ver que outro algoritmo é melhor),
        #é uma tentativa de agrupamento que não ficou muito bom
        #grupos = []
        #visitados = set()

        #limite = 0.60

        #for i in range(len(df_indicador)):

            #if i in visitados:
                #continue

            #grupo = [i]
            #visitados.add(i)

            #for j in range(len(df_indicador)):

                #if j == i:
                    #continue

                #if matriz_sim[i, j] >= limite:

                    #grupo.append(j)
                    #visitados.add(j)

            #grupos.append(grupo)

        grupos = []
        visitados = set()

        limite = 0.60

        for inicio in range(len(df_indicador)):

            if inicio in visitados:
                continue

            grupo = []
            fila = [inicio]

            while fila:

                atual = fila.pop()

                if atual in visitados:
                    continue

                visitados.add(atual)
                grupo.append(atual)

                vizinhos = []

                for j in range(len(df_indicador)):

                    if matriz_sim[atual, j] >= limite:
                        vizinhos.append(j)

                fila.extend(vizinhos)

            grupos.append(grupo)

        grupos = sorted(grupos, key=len, reverse=True)

        for n, grupo in enumerate(grupos[:25], start=1):

            melhor_idx = None
            melhor_score = -1

            for candidato in grupo:
                score_total = sum(
                    matriz_sim[candidato, outro]
                    for outro in grupo
                )

                if score_total > melhor_score:

                    melhor_score = score_total
                    melhor_idx = candidato

            exemplo = df_indicador.iloc[melhor_idx]["Fato"]
            col_a, col_b = st.columns([6, 1])

            with col_a:

                st.markdown(f"""**Grupo {n}** • {len(grupo)} ocorrências. **Exemplo:** {exemplo}""")

            with col_b:

                if st.button("📊 Ver análise", key=f"grupo_{n}"):

                    st.session_state.grupo_selecionado = (df_indicador.iloc[grupo].copy())
                    st.session_state.grupo_confirmado = (df_indicador.iloc[grupo].copy())
                    st.session_state.titulo_grupo = exemplo
                    st.session_state.filtros_grupo = (st.session_state.filtros_analise.copy())
                    st.session_state.pagina_analise = "grupo"
                    st.session_state.checkbox_grupo = {pos: True
                        for pos in range(len(grupo))}

                    st.rerun()

            st.divider()

if (pagina_menu == "📊 Análise de Indicadores" and st.session_state.pagina_analise == "grupo"):

    if st.button("⬅ Voltar para os grupos"):

        filtros = st.session_state.filtros_grupo

        st.session_state.linha_analise = (filtros["linha"])

        st.session_state.maq_analise = (filtros["maq"])

        st.session_state.turno_analise = (filtros["turno"])

        st.session_state.periodo_analise = (filtros["periodo"])

        st.session_state.indicador_sel_analise = (filtros["indicador"])

        st.session_state.grupo_selecionado = None

        st.session_state.titulo_grupo = ""

        st.session_state.pagina_analise = "lista"

        st.rerun()


    st.header("📊 Análise do Grupo")

    st.subheader(
        st.session_state.titulo_grupo
    )

    df_grupo = st.session_state.grupo_confirmado

    datas_ordenadas = (
        df_grupo["Data"]
        .dropna()
        .sort_values()
    )

    intervalos = datas_ordenadas.diff().dt.total_seconds() / 86400

    intervalos = intervalos.dropna()

    if len(intervalos) > 0:

        tempo_medio_entre_ocorrencias = round(
            intervalos.mean(),
            1
        )

    else:

        tempo_medio_entre_ocorrencias = "-"

    tempos_validos = pd.to_numeric(
        df_grupo["Tempo de impacto"],
        errors="coerce"
    ).fillna(0)

    tempo_total = tempos_validos.sum()

    tempo_medio = (
        round(tempos_validos.mean(), 1)
        if len(tempos_validos) > 0
        else 0
    )

    data_max = df_grupo["Data"].max()

    ultimos_30 = df_grupo[df_grupo["Data"] >= (data_max - timedelta(days=30))]

    anteriores_30 = df_grupo[(df_grupo["Data"] < (data_max - timedelta(days=30)))
        &
        (df_grupo["Data"] >= (data_max - timedelta(days=60)))]

    qtd_ultimos = len(ultimos_30)

    qtd_anteriores = len(anteriores_30)

    if qtd_anteriores > 0:

        tendencia_pct = round(
            (
                (qtd_ultimos - qtd_anteriores)
                / qtd_anteriores
            ) * 100,
            1
        )

    else:

        tendencia_pct = None

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Ocorrências",
            len(df_grupo)
        )

    with col2:

        st.metric(
            "Tempo Total",
            f"{int(tempo_total)} min"
        )


    with col3:

        st.metric(
            "Tempo Médio",
            f"{tempo_medio} min"
        )


    with col4:

        st.metric(
            "Linhas Afetadas",
            df_grupo["Linha"].nunique()
        )

    with col5:

        maquinas_unicas = set()

        for lista in df_grupo["Maquina"]:
            maquinas_unicas.update(lista)

        st.metric(
            "Máquinas Afetadas",
            len(maquinas_unicas)
        )


    col_a, col_b, col_c, col_d, col_e = st.columns(5)

    with col_a:

        st.metric(
            "Tempo médio entre ocorrências",
            (
                f"{tempo_medio_entre_ocorrencias} dias"
                if tempo_medio_entre_ocorrencias != "-"
                else "-"
            )
        )

    with col_b:
        if tendencia_pct is not None:

            if tendencia_pct > 0:

                st.metric(
                    "Tendência",
                    f"↗ {tendencia_pct}%"
                )

            elif tendencia_pct < 0:

                st.metric(
                    "Tendência",
                    f"↘ {abs(tendencia_pct)}%"
                )

            else:

                st.metric(
                    "Tendência",
                    "0%"
                )

        else:

            st.metric(
                "Tendência",
                "-"
            )
        st.caption(f"{qtd_ultimos} vs {qtd_anteriores} ocorrências")

    with col_c:
        st.empty()

    with col_d:
        st.empty()

    with col_e:
        st.empty()

    st.markdown("---")
    st.subheader("🏭 Distribuição por Linha")

    df_linhas = df_grupo.copy()
    df_linhas["Tempo_Num"] = pd.to_numeric(df_linhas["Tempo de impacto"],errors="coerce").fillna(0)
    resumo_linhas = (df_linhas.groupby("Linha").agg(
        Ocorrencias=("Linha", "count"),
        Tempo_Total=("Tempo_Num", "sum")).sort_values("Ocorrencias",ascending=False))

    total_ocorrencias = resumo_linhas["Ocorrencias"].sum()

    total_tempo = resumo_linhas["Tempo_Total"].sum()

    resumo_linhas["% Ocorrências"] = (
        resumo_linhas["Ocorrencias"]
        / total_ocorrencias
        * 100
    ).round(1)

    resumo_linhas["% Tempo"] = (
        resumo_linhas["Tempo_Total"]
        / total_tempo
        * 100
    ).round(1)

    resumo_linhas["% Acum. Ocorrências"] = (
        resumo_linhas["% Ocorrências"]
        .cumsum()
        .round(1)
    )

    resumo_linhas["% Acum. Tempo"] = (
        resumo_linhas["% Tempo"]
        .cumsum()
        .round(1)
    )

    resumo_linhas = resumo_linhas[
        [
            "Ocorrencias",
            "% Ocorrências",
            "% Acum. Ocorrências",
            "Tempo_Total",
            "% Tempo",
            "% Acum. Tempo"
        ]
    ]

    resumo_linhas = resumo_linhas.rename(
            columns={
                "Tempo_Total": "Tempo Total (min)",
                "Ocorrencias": "Ocorrências"
            }
        )

    st.dataframe(resumo_linhas, use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Distribuição por Máquina")

    df_maquinas = df_grupo.copy()
    df_maquinas["Tempo_Num"] = pd.to_numeric(df_maquinas["Tempo de impacto"],errors="coerce").fillna(0)
    df_maquinas = df_maquinas.explode("Maquina")
    resumo_maquinas = (df_maquinas.groupby("Maquina").agg(
        Ocorrencias=("Maquina", "count"),
        Tempo_Total=("Tempo_Num", "sum")).sort_values("Ocorrencias",ascending=False))

    total_ocorrencias = resumo_maquinas["Ocorrencias"].sum()

    total_tempo = resumo_maquinas["Tempo_Total"].sum()

    resumo_maquinas["% Ocorrências"] = (
        resumo_maquinas["Ocorrencias"]
        / total_ocorrencias
        * 100
    ).round(1)

    resumo_maquinas["% Tempo"] = (
        resumo_maquinas["Tempo_Total"]
        / total_tempo
        * 100
    ).round(1)

    resumo_maquinas["% Acum. Ocorrências"] = (
        resumo_maquinas["% Ocorrências"]
        .cumsum()
        .round(1)
    )

    resumo_maquinas["% Acum. Tempo"] = (
        resumo_maquinas["% Tempo"]
        .cumsum()
        .round(1)
    )

    resumo_maquinas = resumo_maquinas[
        [
            "Ocorrencias",
            "% Ocorrências",
            "% Acum. Ocorrências",
            "Tempo_Total",
            "% Tempo",
            "% Acum. Tempo"
        ]
    ]

    resumo_maquinas = resumo_maquinas.rename(
        columns={
            "Tempo_Total": "Tempo Total (min)",
            "Ocorrencias": "Ocorrências"
        }
    )

    st.dataframe(resumo_maquinas,use_container_width=True)
    st.caption("* Uma mesma ocorrência pode estar associada a mais de uma máquina.")

    st.markdown("---")
    st.subheader("📈 Ocorrências ao longo do tempo")

    df_temp = pd.DataFrame({"Data": pd.to_datetime(df_grupo["Data"]),
                            "Linha": df_grupo["Linha"],
                            "Fato": df_grupo["Fato"],
                            "Turno": df_grupo["TURNO"]})

    np.random.seed(42)

    df_temp["y"] = np.random.uniform(
        0.49,
        0.51,
        len(df_temp)
    )

    grafico = alt.Chart(df_temp).mark_circle(size=100).encode(
            x=alt.X(
                "Data:T",
                title="Data",
                axis=alt.Axis(
                    format="%d/%m/%y"
                )
            ),

            y=alt.Y(
                "y:Q",
                title="",
                axis=None,
                scale=alt.Scale(
                    domain=[0.45, 0.55]
                )
            ),

            color=alt.value("#4cc9f0"),

            tooltip=[
                alt.Tooltip(
                    "Data:T",
                    title="Data",
                    format="%d/%m/%Y"
                ),
                alt.Tooltip("Linha:N",title="Linha"),
                alt.Tooltip("Turno:N",title="Turno"),
                alt.Tooltip("Fato:N",title="Fato")
            ]
        ).properties(height=180)

    st.altair_chart(grafico,use_container_width=True)
        
    st.markdown("---")
    st.subheader("🔵 Análise por turno")

    turnos_data = []
    for _, row in df_grupo.iterrows():

        tempo = row.get(
            "Tempo de impacto",
            None
        )

        try:
            tempo = float(tempo)
        except:
            tempo = None

        turnos_data.append({
            "Turno": row.get(
                "TURNO",
                "N/A"
            ),
            "Tempo": tempo
        })

    df_turnos = pd.DataFrame(turnos_data)
    df_turnos_validos = df_turnos[df_turnos["Tempo"] > 0]
    zeros_turno = (len(df_turnos) - len(df_turnos_validos))

    turnos_ordem = [
        "1° Turno",
        "2° Turno",
        "3° Turno"
    ]

    ocorrencias_turno = (df_turnos["Turno"].value_counts())
    media_turno = (df_turnos_validos.groupby("Turno")["Tempo"].mean())
    tempo_total_turno = (df_turnos_validos.groupby("Turno")["Tempo"].sum())
    
    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📊 Ocorrências por turno")

        for t in turnos_ordem:

            valor = ocorrencias_turno.get(t,0)

            st.write(f"{t}: {valor}")

    with col2:

        st.markdown("### ⏱ Tempo médio")

        if zeros_turno > 0:
                st.caption(f"⚠️ {zeros_turno} ocorrências com tempo 0 foram desconsideradas")

        for t in turnos_ordem:
            valor = media_turno.get(t, None)

            if pd.notna(valor):
                st.write(f"{t}: {round(valor,1)} min")

            else:
                st.write(f"{t}: sem dados")

    with col3:

        st.markdown("### 🕒 Tempo total")

        for t in turnos_ordem:
            valor = tempo_total_turno.get(t, None)

            if pd.notna(valor):
                st.write(f"{t}: {int(valor)} min")

            else:
                st.write(f"{t}: sem dados")

    st.markdown("---")
    st.subheader("📝 Ocorrências consideradas")

    if st.button("✅ Aplicar seleção"):
        ocorrencias_confirmadas = []

        for pos, (_, row) in enumerate(st.session_state.grupo_selecionado.iterrows()):

            st.session_state.checkbox_grupo[pos] = (
                st.session_state.get(f"grupo_ocorrencia_{pos}", True))

        for pos, (_, row) in enumerate(st.session_state.grupo_selecionado.iterrows()):
            considerar = st.session_state.get(f"grupo_ocorrencia_{pos}", True)
            if considerar:
                ocorrencias_confirmadas.append(row)

        if len(ocorrencias_confirmadas) > 0:

            st.session_state.grupo_confirmado = (
                pd.DataFrame(
                    ocorrencias_confirmadas
                )
            )

            st.toast("✅ Seleção aplicada!")
            st.rerun()

    total_original = len(st.session_state.grupo_selecionado)
    total_confirmado = len(st.session_state.grupo_confirmado)
    st.caption(f"Ocorrências consideradas: {total_confirmado} de {total_original}")

    for pos, (_, row) in enumerate(st.session_state.grupo_selecionado.iterrows()):
        col_check, col_texto = st.columns([1, 10])

        with col_check:

            considerar = st.checkbox("",value=st.session_state.checkbox_grupo.get(pos,True), 
                                     key=f"grupo_ocorrencia_{pos}")

        with col_texto:
            tempo = row.get("Tempo de impacto", "-")
            maquinas = ", ".join(row["Maquina"]) if row["Maquina"] else "-"
            preview = (row["Fato"][:80] + "..." if len(row["Fato"]) > 80 else row["Fato"])

            icone = "🟢" if considerar else "🔴"

            with st.expander(
                f"{icone} 🔎 "
                f"{row['Data'].strftime('%d/%m/%Y')} | "
                f"⏱ {tempo} min | "
                f"{preview}"
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**📝 Fato:**")
                    st.write(row["Fato"] if row["Fato"]  else "-")
                    st.markdown("**🔍 Causa:**")
                    st.write( row["Causa"] if row["Causa"] else "-")
                    st.markdown("**⚙️ Ação:**")
                    st.write(row["Ação"] if row["Ação"] else "-")

                with col2:
                    st.markdown( f"📍 **{row['Linha']} | {maquinas}**" )
                    st.write(row["TURNO"])
                    st.write(f"⏱ {tempo} min")

