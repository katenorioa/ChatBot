import streamlit as st
import os
import tempfile
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(
    page_title="Asistente Académico - Curso IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>

/* ===== FONDO GLOBAL ===== */

html, body, .stApp {
    background-color: #ffffff !important;
    color: #1e293b !important;
}

/* CONTENEDOR PRINCIPAL */

[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}

/* CONTENEDOR CENTRAL */

.main .block-container {
    background-color: #ffffff !important;
    color: #1e293b !important;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
}

[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* ===== TEXTOS ===== */

h1, h2, h3, h4 {
    color: #1e293b !important;
}

p, span, label, div {
    color: #334155 !important;
}

/* ===== INPUT CHAT ===== */

[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border-top: 1px solid #e2e8f0 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
}

/* INPUT GENERAL */

input {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
}

/* ===== MENSAJES CHAT ===== */

[data-testid="stChatMessage"] {
    background-color: #f8fafc !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
}

[data-testid="stChatMessage"] * {
    color: #1e293b !important;
}

/* MENSAJE USUARIO */

[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #dbeafe !important;
}

/* MENSAJE BOT */

[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: #f1f5f9 !important;
}

/* ===== BOTONES ===== */

button {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
}

button:hover {
    background-color: #1d4ed8 !important;
}

/* ===== FILE UPLOADER ===== */

[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
}

/* ===== TARJETAS ===== */

.info-card {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}

.welcome-card {
    background-color: #ffffff !important;
}

/* ===== ELIMINAR ZONAS OSCURAS ===== */

section {
    background-color: #ffffff !important;
}

header {
    background-color: #ffffff !important;
}

footer {
    background-color: #ffffff !important;
}

/* ===== CONTENEDOR INFERIOR (CHAT BAR) ===== */

[data-testid="stBottomBlockContainer"] {
    background-color: #ffffff !important;
    border-top: 1px solid #e2e8f0 !important;
}

/* CONTENEDOR DEL INPUT */

[data-testid="stChatInputContainer"] {
    background-color: #ffffff !important;
}

/* INPUT COMPLETO */

[data-testid="stChatInput"] {
    background-color: #ffffff !important;
}

/* AREA DEL TEXTAREA */

[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
}

/* CONTENEDOR EXTERNO DEL INPUT */

.stChatFloatingInputContainer {
    background-color: #ffffff !important;
}

/* ELIMINA FONDO OSCURO GLOBAL INFERIOR */

footer {
    background-color: #ffffff !important;
}

/* TEXTO DEL PLACEHOLDER (Ej: ¿Qué temas aborda el Módulo 1?) */

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;  /* gris visible */
    opacity: 1 !important;
}

/* Para compatibilidad completa */

textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

textarea::-webkit-input-placeholder {
    color: #64748b !important;
}

textarea::-moz-placeholder {
    color: #64748b !important;
}

textarea:-ms-input-placeholder {
    color: #64748b !important;
}
</style>
""", unsafe_allow_html=True)



# Constantes
CARPETA_DB = "chroma_db_uce"
LOGO_PATH = "logo_uce.png"
AVATAR_PATH = "Dante.png"

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ Falta la API Key. Configúrala en Streamlit Cloud.")
    st.stop()

# --- MOTOR DE INTELIGENCIA ARTIFICIAL ---
def get_llm():
    """
    Configuración del modelo Groq para entorno académico.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=GROQ_API_KEY
    )

def get_embeddings():
    """
    Configuración de embeddings con HuggingFace.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# --- PROCESAMIENTO DE DOCUMENTOS ---
def procesar_pdfs(archivos_pdf):
    # Limpieza preventiva
    if os.path.exists(CARPETA_DB):
        try:
            shutil.rmtree(CARPETA_DB)
        except Exception as e:
            st.error(f"Error limpiando caché: {e}")

    documentos = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    progreso = st.progress(0, text="Procesando módulos del curso...")
    
    for i, archivo in enumerate(archivos_pdf):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(archivo.read())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            documentos.extend(docs)
        except Exception as e:
            st.error(f"Error en archivo {archivo.name}: {e}")
        finally:
            os.remove(tmp_path)
        
        progreso.progress(int((i + 1) / len(archivos_pdf) * 50))

    splits = text_splitter.split_documents(documentos)
    
    if splits:
        progreso.progress(70, text="Generando base de conocimiento...")
        Chroma.from_documents(
            documents=splits,
            embedding=get_embeddings(),
            persist_directory=CARPETA_DB
        )
        progreso.progress(100, text="Carga completada exitosamente.")
        return True
    return False

# --- INTERFAZ DE USUARIO ---

# Sidebar: Navegación y Datos Institucionales
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.warning("Falta imagen 'logo_uce.png'")
    
    st.markdown("---")
    st.header("Menú Principal")
    opcion = st.radio("Seleccione Perfil:", ["Estudiante (Consultas)", "Docente (Administración)"])
    
    # Botón para limpiar chat (Solo visible en modo estudiante)
    if opcion == "Estudiante (Consultas)":
        if st.button("Limpiar Conversación"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    
    # Información Académica
    st.markdown("### Información Académica")
    st.markdown("""
    **Facultad:** Ingeniería  
    **Carrera:** Ingeniería en Sistemas  
    **Materia:** Sistemas Operativos 2
    """)
    
    st.markdown("---")
    
    # Equipo Desarrollador
    st.markdown("### Equipo Desarrollador")
    st.markdown("""
    - Daniel Quiros
    - Cristopher Alquinga
    - Milisen Narváez
    - Kevin Tenorio
    - Marlon Argoti
    """)
    
    st.markdown("---")
    
    # Algoritmos Implementados
    st.markdown("### Algoritmos Implementados")
    st.markdown("""
    **Regresión Lineal**  
    Análisis de patrones en consultas para identificar tendencias y mejorar respuestas basándose en el historial de interacciones.
    
    **RAG (Retrieval-Augmented Generation)**  
    Utiliza similitud de coseno en embeddings vectoriales para recuperar información precisa de los documentos PDF.
    
    **Análisis de Embeddings**  
    Transformación de texto en vectores numéricos de alta dimensión para comprensión semántica contextual.
    """)
    
    st.markdown("---")
    st.markdown('<p class="caption">Facultad de Ingeniería - UCE 2026</p>', unsafe_allow_html=True)

# --- VISTA: ADMINISTRACIÓN (DOCENTE) ---
if opcion == "Docente (Administración)":
    # Custom Header for Docent Panel
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(135deg, #a5b4fc 0%, #c7d2fe 100%); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   background-clip: text;
                   font-size: 2.2rem;
                   font-weight: 800;
                   margin-bottom: 0.5rem;
                   animation: fadeInUp 0.6s ease-out;">
            Panel Docente
        </h1>
        <p style="color: #94a3b8; font-size: 1rem;">
            Gestión de Módulos del Curso
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Cargue los sílabos y módulos PDF para actualizar el conocimiento del asistente.")
    
    password = st.text_input("Clave de Acceso:", type="password")
    
    if password == "uce2026":
        st.warning("⚠️ IMPORTANTE: Si cambió de Ollama a Groq, debe limpiar la base de datos antigua primero.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Limpiar Base de Datos", type="secondary"):
                if os.path.exists(CARPETA_DB):
                    try:
                        shutil.rmtree(CARPETA_DB)
                        st.success("Base de datos eliminada. Ahora puede cargar nuevos PDFs.")
                    except Exception as e:
                        st.error(f"Error al limpiar: {str(e)}")
                else:
                    st.info("No hay base de datos para limpiar.")
        
        with col2:
            db_status = "✅ Existe" if os.path.exists(CARPETA_DB) else "❌ No existe"
            st.metric("Estado BD", db_status)
        
        st.markdown("---")
        
        uploaded_files = st.file_uploader("Subir Archivos PDF", type="pdf", accept_multiple_files=True)
        
        if st.button("Procesar y Actualizar IA", type="primary"):
            if uploaded_files:
                with st.spinner("Analizando documentos..."):
                    procesar_pdfs(uploaded_files)
                st.success("Base de datos actualizada correctamente.")
            else:
                st.error("Debe seleccionar al menos un archivo.")

# --- VISTA: ESTUDIANTE (CHAT) ---
elif opcion == "Estudiante (Consultas)":
    # Tarjeta de Bienvenida con Avatar
    if os.path.exists(AVATAR_PATH):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(AVATAR_PATH, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 1rem 0;">
                <h1 style="background: linear-gradient(135deg, #a5b4fc 0%, #c7d2fe 100%); 
                           -webkit-background-clip: text; 
                           -webkit-text-fill-color: transparent; 
                           background-clip: text;
                           font-size: 2rem;
                           font-weight: 800;
                           margin-bottom: 0.5rem;">
                    Hola, soy Lumen
                </h1>
                <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6; margin: 0;">
                    Tu asistente virtual inteligente para el curso de IA. 
                    Estoy aquí para ayudarte a resolver dudas sobre los módulos del curso.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ej: ¿Qué temas aborda el Módulo 1?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not os.path.exists(CARPETA_DB):
                st.info("El docente aún no ha cargado los módulos del curso.")
                st.stop()
            
            with st.spinner("Consultando bibliografía..."):
                try:
                    vectorstore = Chroma(
                        persist_directory=CARPETA_DB, 
                        embedding_function=get_embeddings()
                    )
                    
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
                    docs = retriever.invoke(prompt)
                    contexto = "\n\n".join([d.page_content for d in docs])

                    prompt_sistema = f"""
                    Rol: Eres el Asistente Docente oficial del Curso de IA de la Universidad Central del Ecuador.
                    
                    Instrucciones:
                    1. Responde ÚNICAMENTE basándote en el siguiente contexto de los módulos.
                    2. Sé formal, motivador y preciso.
                    3. Usa viñetas para enumerar temas o pasos.
                    4. Si la información no existe en los documentos, indícalo claramente.
                    
                    
                    Contexto:
                    {contexto}

                    Pregunta del Estudiante:
                    {prompt}
                    """
                    
                    llm = get_llm()
                    stream = llm.stream(prompt_sistema)
                    
                    # MEJORA 2: Capturar la respuesta real para el historial
                    response = st.write_stream(stream)
                    
                except Exception as e:
                    st.error(f"Error al generar respuesta: {str(e)}")
                    response = "Error técnico."
        
        # Guardamos la respuesta REAL en el historial
        st.session_state.messages.append({"role": "assistant", "content": response})
