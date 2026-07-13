#Versão 1.2
#última alteração no código 06/07/2026
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import ast

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="App Ata Lider", layout="wide")
st.title("📊 Monitor de Ocorrências Teste online")

# ==============================
# STATE (NAVEGAÇÃO)
# ==============================

if "pagina" not in st.session_state:
    st.session_state.pagina = "lista"

if "idx_selecionado" not in st.session_state:
    st.session_state.idx_selecionado = None

if "pagina_cards" not in st.session_state:
    st.session_state.pagina_cards = 0


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
    #similaridade = cosine_similarity(matriz) #DEVO TIRAR ISSO DAQUI?

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

    df["Fato"] = df["Fato"].apply(clean_text)
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

textos = df["Fato"].fillna("").tolist()
vectorizer, matriz = calcular_modelo(textos)


# ==============================
# FILTROS
# ==============================

if st.session_state.pagina == "lista":

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
        query_limpa = clean_text(query)
        query_vec = vectorizer.transform([query_limpa])

        # similaridade com todos os registros
        scores = cosine_similarity(query_vec, matriz)[0]

        # adiciona score ao dataframe
        df_filtrado = df_filtrado.copy()
        df_filtrado["score"] = scores[df_filtrado.index]

        # boost para match exato
        df_filtrado["score"] += df_filtrado["Fato"].apply(
            lambda x: 0.2 if query.lower() in x else 0
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

if st.session_state.pagina == "lista":
    
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
        indicador_texto = ", ".join(row["INDICADOR FORA"])

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
                    st.session_state.idx_selecionado = idx
                    st.session_state.pagina = "detalhe"
                    st.rerun()

            st.divider()

    mostrar_paginacao(total_paginas, "bottom")
# ==============================
# 🔵 PAGINA 2 — DETALHE
# ==============================

if st.session_state.pagina == "detalhe":

    idx = st.session_state.idx_selecionado
    row = df.loc[idx]

    # 🔙 VOLTAR
    if st.button("⬅️ Voltar"):
        st.session_state.pagina = "lista"
        st.rerun()

    st.markdown("## 📌 Detalhe da Ocorrência")

    maquinas_texto = ", ".join(row["Maquina"])
    indicador_texto = ", ".join(row["INDICADOR FORA"])

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
    indices_similares = [i for i in indices_similares if i != idx][:100]

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
        if similares[i] < 0.25:
            continue

        lista_similares.append((i, similares[i]))

    # ==============================
    # 📊 MÉTRICAS
    # ==============================

    total_similares = len(lista_similares)

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
        for i, score in lista_similares[:10]:

            outra = df.iloc[i]
            score = round(score, 2)

            maquinas_out = ", ".join(outra["Maquina"]) if outra["Maquina"] else "-"

            
            # Criar preview do fato
            preview = outra["Fato"][:80] + "..." if len(outra["Fato"]) > 80 else outra["Fato"]

            # Tempo de impacto no título
            tempo = outra.get("Tempo de impacto", "-")

            with st.expander(
                f"🔎 Similar ({score}) | {outra['Data'].strftime('%d/%m/%Y')} | ⏱ {tempo} min | {preview}"
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

        if total_similares > 0:

            datas = []
            tipos = []

            # SIMILARES
            for i, score in lista_similares:
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

        for i, score in lista_similares:
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

        # similares
        for i, score in lista_similares:
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