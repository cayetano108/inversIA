from dotenv import load_dotenv

load_dotenv()

from langchain_deepseek import ChatDeepSeek

# import tensorflow as tf
import os

import json

# import time
from typing import Literal

import time

from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic.v1 import BaseModel, Field

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

# print("Is GPU available?:", tf.config.list_physical_devices("GPU"))

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY_EMBEDDING = os.getenv("OPENAI_API_KEY_EMBEDDING")



# LLMs definition

deepseek = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=DEEPSEEK_API_KEY,
    # other params...
)

import langchain_openai as langchain_openai
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

openai = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    openai_api_key=OPENAI_API_KEY_EMBEDDING,
)

openai_gpt4o = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    openai_api_key=OPENAI_API_KEY_EMBEDDING,
)


gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # también tienes "gemini-2.5-pro"
    temperature=0,
    max_output_tokens=None,
    google_api_key=os.environ["GOOGLE_AI_API_KEY"],
)
# llms = [openai]
# llm = openai

llms = [deepseek]
llm = deepseek

# llms = [gemini]
# llm = gemini

############     ROUTER     ############


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "redirect"] = Field(
        ...,
        description="Given a user question choose to route it to redirect or a vectorstore.",
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """Eres un experto en dirigir una pregunta de usuario hacia "vectorstore" o "redirect".

Tu tarea es decidir si la información proporcionada debe resolverse con el conocimiento del corpus (vectorstore) o si necesita redirigir la conversacion.

Vas a redirigir la pregunta a "vectorstore" si está relacionada con conceptos de inversión, tales como:
- planificación financiera
- renta variable
- interés compuesto
- fondos indexados o ETFs
- fiscalidad de productos financieros (fondos, ETFs, acciones)
- diversificación de cartera
- estrategias como el DCA
- errores comunes al invertir
- elección de productos financieros
- decisiones de consumo que afecten al patrimonio (ej: comprar coche, vivienda)
- juegos de azar, seguros, criptomonedas o formación financiera
- Herencia de inversiones y su impacto en el patrimonio
- Información del mercado financiero que pueda afectar a decisiones de inversión (ej: tipos de interés, inflación, información de acciones o fondos, datos de empresas, etc.)

Ten también en cuenta que si el usuario te pregunta precios de cualquier producto financiero, información de un fondo o ETF, o datos históricos de una acción, puedas acceder a ellos más adelante a través de herramientas que te proporcionaré, por lo que en esos casos también es preferible que redirijas a "vectorstore" para aprovechar esa información, siempre que la pregunta esté relacionada con inversión o finanzas personales.
También puedes tener en cuenta el historial de la conversación: si las preguntas anteriores estaban relacionadas con inversión, es más probable que esta también lo esté, incluso si la formulación actual es ambigua.


IMPORTANTE:
- Tu salida debe ser solamente: "vectorstore" o "redirect".
"""


route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


#############     DOCUMENT LOADER     ############
##### PRUEBAS INICIALES CON DOCUMENTOS PREVIOS

# archivo_txt = "./finance_data/self/guia_completa_inversion.txt"

# # Cargar el archivo de texto

# with open(archivo_txt, "r", encoding="utf-8") as file:
#     contenido_txt = file.read()

# contenido_txt

# # divide el txt en partes más pequeñas, split por '#############################################'

# docs_list = contenido_txt.split("#############################################")
# print("numero de documentos: ", len(docs_list))
# docs_list = [doc for doc in docs_list if len(doc) > 0]

# docs_list = [{"page_content": doc, "metadata": {"summary": None}} for doc in docs_list]


###########    VERSION FINAL PARA LIBROS Y GUIAS
archivo_txt = "./finance_data/cleaned_books/full_book_guides.txt"

with open(archivo_txt, "r", encoding="utf-8") as file:
    contenido_txt = file.read()

# Split por tamaño de caracteres, respetando saltos de párrafo
CHUNK_SIZE = 1200  # ajusta si quieres chunks más grandes o pequeños


def split_by_chars(text, chunk_size):
    chunks = []
    while len(text) > chunk_size:
        # Buscar el último \n\n antes del límite para no cortar a mitad de párrafo
        split_at = text.rfind("\n\n", 0, chunk_size)
        if split_at == -1:  # si no hay párrafo, cortar en el límite
            split_at = chunk_size
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text.strip())
    return chunks


docs_list = split_by_chars(contenido_txt, CHUNK_SIZE)
docs_list = [doc for doc in docs_list if len(doc) > 0]

print("numero de documentos: ", len(docs_list))

docs_list = [{"page_content": doc, "metadata": {"summary": None}} for doc in docs_list]


# ############## SUMMARIZARION    ##############
## COMENTADO PARA NO VOLVER A RESUMIR LOS DOCUMENTOS, YA QUE SE HAN RESUMIDO Y GUARDADO EN summary_metadata_deepseek_BOOKGUIDES.json

# # Prompt
# system = """Estás asumiendo el papel de un experto en inversión a largo plazo y educación financiera básica.
# Necesito que hagas un resumen extremadamente pequeño, a modo de explicación, de la información que voy a pasarte.
# Ten en cuenta que la información o resumen que proporciones va a ser utilizada como metadata para los documentos que
# nutrirán el contexto de un sistema RAG, por tanto, intenta que el resumen sea lo más conciso y útil
# a la hora de realizar esa búsqueda para encontrar el contexto adecuado de entre todos los documentos,
# en base a la futura pregunta del usuario.

# Necesito que tu respuesta sea directamente lo que mejor consideres según las especificaciones que te he dado,
# sin incluir ninguna introducción. Simplemente el texto de respuesta.
# A continuación, te pasaré la información del documento."""

# deepseek_metadata = []

# for cont, doc in enumerate(docs_list):
#     messages = [
#         ("system", system),
#         ("human", doc["page_content"]),
#     ]

#     t1 = time.time()
#     answer = llm.invoke(messages)
#     print("Respuesta --> ", answer)
#     t2 = time.time()
#     deepseek_metadata.append(answer.content)
#     print(f"Documento {cont+1} procesado en {t2-t1} segundos \n")

# # # Guardar los metadatos en un archivo json para su uso posterior


# # with open("finance_data/self/summary_metadata_deepseek_NEWCONTEXT.json", "w") as f:
# #     json.dump(deepseek_metadata, f)

# with open(
#     "finance_data/cleaned_books/summary_metadata_deepseek_BOOKGUIDES.json", "w"
# ) as f:
#     json.dump(deepseek_metadata, f)


# ############# LOAD SUMMARIZED METADATA #############
# AQUI ES DONDE SE CARGAN LOS METADATOS YA RESUMIDOS PARA CADA DOCUMENTO, PARA QUE SEAN UTILIZADOS COMO CONTEXTO EN EL RAG

# with open(
#     "finance_data/cleaned_books/summary_metadata_deepseek_BOOKGUIDES.json", "r"
# ) as f:
#     summary_metadata_deepseek = json.load(f)


# # añadir el contenido adicional a los metadatos de cada documento

# for idx, doc in enumerate(docs_list):
#     doc["metadata"]["summary"] = summary_metadata_deepseek[idx]

# # documentos compatibles con langchain
# docs_list = [
#     Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_list
# ]


#############     RETRIEVERS & VECTORSTORE     #############
## ESTA PARTE ESTÁ COMENTADA PARA NO VOLVER A CREAR LOS VECTORSTORES, YA QUE SE HAN CREADO Y GUARDADO EN DISCO,
## MANTENGO EL CÓDIGO PARA MOSTRAR CÓMO SE HAN CREADO Y POR SI SE QUIERE VOLVER A CREAR CON OTRAS COMBINACIONES DE chunk_size y chunk_overlap


# chunk_sizes = [100, 500]
# chunk_overlaps = [10, 25]

# chunk_sizes = [500]
# chunk_overlaps = [50]

# combinations = [(size, overlap) for size in chunk_sizes for overlap in chunk_overlaps]

# retrievers = []

# combinations = [
#     # --- Fragmentación fina ---
#     (100, 10),  # microchunks casi sin solapamiento
#     (100, 25),  # fino con más solapamiento
#     (100, 50),  # fino con mucho solapamiento
#     (250, 25),  # tamaño medio-fino
#     (250, 50),
#     # --- Fragmentación media ---
#     (500, 50),  # bastante usado (default en muchos ejemplos de RAG)
#     (500, 100),  # solapamiento ~20 %
#     (500, 200),  # solapamiento ~40 %
#     # --- Fragmentación gruesa ---
#     (800, 80),  # ~10 % solapamiento
#     (800, 160),  # ~20 %
#     (800, 400),  # ~50 %
#     # --- Máximos ---
#     (1000, 100),  # grandes bloques con poco solapamiento
#     (1000, 200),
# ]

# ##### COMENTADO PARA NO VOLVER A CREAR LOS VECTORSTORES ######
# print("Creando los nuevos vectorstores...")
# # for size in chunk_sizes:
# #     for overlap in chunk_overlaps:

# for comb in combinations:

#     size, overlap = comb

#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=size, chunk_overlap=overlap
#     )
#     doc_splits = text_splitter.split_documents(docs_list)

#     print(f"Number of document splits: {len(doc_splits)}")

#     # Generar embeddings por separado
#     # embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")  # textembedding-gecko@003
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-large", api_key=OPENAI_API_KEY_EMBEDDING
#     )

# comentado despues de crear nueva forma de resumenes
# embedding_vectors = embeddings.embed_documents(
#     [doc.page_content for doc in doc_splits]
# )

# print(f"Number of embeddings: {len(embedding_vectors)}")

# # Verificar si las longitudes coinciden
# if len(doc_splits) != len(embedding_vectors):
#     print("Discrepancia entre la cantidad de documentos y embeddings.")
#     # Opcional: depurar qué documentos no se están procesando correctamente
#     for i, (doc, embedding) in enumerate(zip(doc_splits, embedding_vectors)):
#         if not embedding:
#             print(f"Documento sin embedding: {doc.page_content[:100]}...")

# BATCH_SIZE = 200
# texts = [
#     f"{doc.metadata.get('summary', '')}\n\n{doc.page_content}" for doc in doc_splits
# ]

# vectorstore = None
# for i in range(0, len(texts), BATCH_SIZE):
#     batch = texts[i : i + BATCH_SIZE]
#     print(
#         f"  Batch {i // BATCH_SIZE + 1}/{-(-len(texts) // BATCH_SIZE)} ({i + len(batch)}/{len(texts)} chunks)"
#     )
#     if vectorstore is None:
#         vectorstore = FAISS.from_texts(batch, embeddings)
#     else:
#         vectorstore.add_texts(batch)

# retriever = vectorstore.as_retriever()

# # para cada retriever, asignale el nombre de la combinacion de chunk_size y chunk_overlap

# retriever.name = f"BOOKGUIDES_chunk_size_{size}_overlap_{overlap}"

# vectorstore.save_local(f"./finance_data/vectorstores/BOOKGUIDES_{retriever.name}")

# # print(f"Number of documents in vectorstore: {len(retriever)}")
# retrievers.append(retriever)
# print(f"Vectorstore {retriever.name} creado y guardado en disco.")


# Cargar el vectorstore desde el disco, para evitar volver a crear el vectorstore

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=OPENAI_API_KEY_EMBEDDING
)
vectorstore = FAISS.load_local(
    "./finance_data/vectorstores/BOOKGUIDES_chunk_size_500_overlap_50",
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)  # k: número de documentos a recuperar


##### Carga de varios retrievers
# crea el objeto retrievers que contiene la carga de todos los vectorstores de la carpeta
# retrievers = []

# for file in os.listdir("./finance_data/vectorstores/"):
#     if file.endswith(".faiss"):
#         vectorstore = FAISS.load_local(f"./finance_data/vectorstores/{file}", embeddings, allow_dangerous_deserialization=True)
#         retriever = vectorstore.as_retriever()
#         retrievers.append(retriever)


###########   CREACION DE AUGMENTED QUERIES   #############


def generate_augment_query(query, llm):

    system = """Eres un asistente experto en inversión a largo plazo y educación financiera.

    Vas a recibir una pregunta de un usuario sobre temas financieros o de inversión.
    Tu objetivo es añadir información adicional que complemente la pregunta original, 
    con el fin de que posteriormente otro agente pueda responderla de forma más completa.

    Por ejemplo, puedes añadir aclaraciones útiles sobre términos financieros, conceptos de inversión, 
    o señalar qué factores o matices habría que tener en cuenta al formular o responder esa pregunta.

    Por favor, sé claro, conciso y breve. 
    MUY IMPORTANTE: Sé que eres capaz de resolver la pregunta, pero NO LA RESPONDAS, 
    solo debes ampliar su contexto y enriquecerla con frases relevantes.

    Tu respuesta debe repetir la pregunta original tal cual, y en el siguiente párrafo, 
    añadir lo que consideres necesario para contextualizarla mejor. NO modifiques la pregunta original.

    Si una pregunta no tiene relación con la inversión o las finanzas personales, 
    mantén la pregunta igualmente

    A pesar de todo, intenta encontrar un punto de conexión con la inversión o las finanzas personales,
    para que la pregunta pueda ser respondida de forma útil. Si no es posible,
    simplemente aclara que no tiene relación con la temática de inversión.

    Por ejemplo, si la pregunta es "¿Qué es un coche?",
    puedes responder "¿Qué es un coche? \n\n Un coche es un vehículo de motor utilizado para el transporte. Si bien no es un tema de inversión,
    es importante considerar el coste de mantenimiento y la depreciación al invertir en un coche."

    También tienes que tener en cuenta el historial anterior, si existe, y este tiene relación con el contexto dado, dale importancia

    A continuación, te paso la pregunta del usuario:
    """

    # one shot, few shots
    messages = [
        ("system", system),
        ("human", query),
    ]

    return llm.invoke(messages)


#############   DOCUMENT GRADING   #############


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

    # Prompt


system = """Eres un experto en inversión a largo plazo y educación financiera.

Tu tarea es decidir si un documento puede ser útil para responder a una pregunta del usuario, relacionada con finanzas personales, inversión o toma de decisiones económicas.

Debes responder "sí" si el documento:
- Trata temas financieros que puedan **contextualizar** o ayudar a responder la pregunta, aunque no la responda de forma literal.
- Menciona conceptos como: activos que se deprecian, coste de oportunidad, decisiones de consumo, gestión del riesgo, impacto en el patrimonio, endeudamiento, etc.
- Puede orientar al usuario a **no cometer errores comunes** en decisiones financieras.

Solo responde "no" si el documento **no tiene ninguna relación útil** con finanzas, economía o inversión (por ejemplo, si habla de comida, deporte, historia, etc.).

Ejemplos:

Pregunta: ¿Voy a comprarme un coche deportivo, es una buena inversión?
Documento: Los activos que se deprecian como los coches pierden valor desde que se compran y rara vez generan ingresos.
Respuesta: sí

Pregunta: ¿Qué fondo indexado me conviene?
Documento: El interés compuesto es clave para entender el crecimiento de las inversiones.
Respuesta: sí

Pregunta: ¿Es buena idea tener muchos dividendos?
Documento: La fiscalidad de los dividendos puede reducir su rentabilidad neta.
Respuesta: sí

Pregunta: ¿Qué ventajas fiscales tienen los fondos de inversión?
Documento: La vitamina C refuerza el sistema inmunológico.
Respuesta: no

Ahora, evalúa el siguiente documento:
"""


structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Documento recuperado: \n\n {document} \n\n Pregunta del usuario: {question}",
        ),
    ]
)

retrieval_grader_relevance = grade_prompt | structured_llm_grader_docs


############   GENERATION   #############

# Prompt
prompt = ChatPromptTemplate.from_template(
    # Debes responder con precisión, aunque no estaría mal del todo un poco de humor en pequeña medida, sobretodo si la pregunta tiene un tono mas informal.
    # Este humor tiene que ser sutil, camuflado, inteligente y no forzado ni excesivo. Evita emoticonos, y evita los puntos suspensivos que pretenden simular una pausa.
    """Eres un asistente para tareas de pregunta y respuesta. Usa las siguientes partes de contexto extraido para responder a la pregunta. 
    Es importante que, aunque te otorgo cierta libertad, intentes no salirte del contexto proporcionado.
    Ten en cuenta el historial de conversación reciente, si existe, y utilízalo para enriquecer tu respuesta, si fuera necesario.
    Recuerda, es importante que no hagas referencia a tu propia existencia como asistente, ni a la de los documentos que has utilizado para responder.
    Por ejemplo, no digas "según el documento", "según el contexto " o "según la información que tengo", simplemente responde a la pregunta del usuario con ese contexto.
Pregunta: {question}
Contexto: {context}
Respuesta:"""
)

# Chain
rag_chain = prompt | llm | StrOutputParser()


##############   HALLUCINATION   #############


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Don't consider calling external APIs for additional information. Answer is supported by the facts, 'yes' or 'no'."
    )


# LLM with function call
structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """Eres un evaluador que está determinando si una respuesta generada por un LLM está respaldada por un conjunto de documentos recuperados. 
Restringe tu calificación a un puntaje binario, ya sea 'sí' o 'no'. 
Si la respuesta está respaldada, parcialmente respaldada o tiene relación con el conjunto de hechos, o incluso con el historial de conversación, califícalo como 'sí'.
No consideres llamar a API externas para obtener información adicional como coherente con los documentos."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination


##############   ANSWER GRADING   #############


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)

# Prompt
system = """Eres un evaluador que determina si una respuesta aborda/resuelve una pregunta. \n
    Da una puntuación binaria 'sí' o 'no'. 'sí' significa que la respuesta ha abordado la pregunta, 'no' significa que no lo ha hecho.
    Por ejemplo, si la pregunta es ¿que es un fondo indexado? y la respuesta es "Un fondo indexado es un tipo de fondo de inversión que busca replicar el rendimiento de un índice específico del mercado",
    entonces la respuesta es "sí". 
    Si la pregunta fuera ¿que es un fondo indexado?, y la respuesta fuera "el tipo de interes es un concepto financiero que se refiere al coste del dinero", la respuesta sería "no".
    A continuación, evalúa la pregunta y la respuesta generada por el LLM."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader_answer


##########################################
#  WORKFLOW
##########################################

from typing_extensions import TypedDict, Annotated
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from pprint import pprint


# STATE
class GraphState(TypedDict):
    question: Annotated[str, "input"]
    augmented_question: str
    generation: str
    redirect: str
    documents: List[Document]
    step_count: int
    chat_history: List[tuple[str, str]]
    source_type: str


# --- UI STATUS HELPER (compat Chainlit / Gradio) ---
def _ui_status(state, text: str):
    """
    Actualiza el mensaje de estado en Chainlit desde nodos síncronos.
    Si no hay UI o Chainlit no está disponible, no hace nada.
    Compatible con APIs antiguas y nuevas de Chainlit.
    """
    try:
        msg = state.get("__ui_status_msg")
        if not msg:
            return
        try:
            import chainlit as cl
        except Exception:
            return

        def _do_update():
            try:
                # API nueva (msg.update(content=...))
                return cl.run_sync(msg.update(content=text))
            except TypeError:
                # API antigua (asigna content y llama update() sin args)
                msg.content = text
                return cl.run_sync(msg.update())

        _do_update()
    except Exception:
        # Nunca rompas el flujo por la UI
        pass


# AUGMENT
def augment(state: GraphState) -> GraphState:
    state["__ui_stage"] = (
        f"🔍 AUMENTANDO LA PREGUNTA — Step {state.get('step_count',0)+1}/10"
    )

    step_count = state.get("step_count", 0) + 1
    print(f"PREGUNTA ORIGINAL: {state['question']}")
    print(f"---AUMENTANDO LA PREGUNTA--- Step {step_count}/10")

    original = state["question"]
    history = state.get("chat_history", [])

    # Construir el texto del historial, si existe
    if history:
        print("Existe historial de conversación, añadiéndolo a la pregunta...")
        history_text = "\n".join([f"{role}: {msg}" for role, msg in history[-6:]])
        input_text = (
            f"Historial anterior:\n{history_text}\n\nPregunta actual:\n{original}"
        )
    else:
        print("No existe historial de conversación, añadiendo solo la pregunta...")
        input_text = original

    # Enviar al modelo para ampliar
    augmented = generate_augment_query(input_text, llm)

    state["augmented_question"] = f"{original}\n\n{augmented.content.strip()}"
    print(f"PREGUNTA AUMENTADA: {state['augmented_question']}")
    state["step_count"] = step_count
    return state



# ROUTER
def route_question(state: GraphState) -> str:
    state["__ui_stage"] = (
        f"🧭 RUTEANDO LA PREGUNTA — Step {state.get('step_count',0)}/10"
    )

    print(f"---RUTEANDO LA PREGUNTA--- Step {state['step_count']}/10")

    # Usa el historial si existe
    history = state.get("chat_history", [])
    current_question = state.get("augmented_question", state["question"])

    if history:
        history_text = "\n".join([f"{role}: {msg}" for role, msg in history[-6:]])
        routing_input = f"Historial reciente:\n{history_text}\n\nPregunta actual:\n{current_question}"
    else:
        routing_input = current_question

    source = question_router.invoke({"question": routing_input})
    print(f"FUENTE DE INFORMACIÓN ELEGIDA PARA ESTA PREGUNTA: {source.datasource}")
    return "redirect" if source.datasource == "redirect" else "vectorstore"


# RETRIEVE
def retrieve(state: GraphState) -> GraphState:
    state["__ui_stage"] = (
        f"📚 RECUPERANDO DOCUMENTOS — Step {state.get('step_count',0)+1}/10"
    )

    step_count = state.get("step_count", 0) + 1
    print(f"---RECUPERANDO DOCUMENTOS--- Step {step_count}/10")

    question = state.get("augmented_question", state["question"])
    history = state.get("chat_history", [])
    history_text = "\n".join(
        [f"{role}: {msg}" for role, msg in history[-6:]]
    )  # últimos turnos

    full_query = f"{history_text}\n\nPregunta actual:\n{question}"
    documents = retriever.invoke(full_query)

    state["documents"] = documents
    state["step_count"] = step_count
    return state


# GRADE DOCUMENTS
def grade_documents(state: GraphState) -> GraphState:
    state["__ui_stage"] = (
        f"🧪 EVALUANDO DOCUMENTOS — Step {state.get('step_count',0)+1}/10"
    )

    step_count = state.get("step_count", 0) + 1
    print(f"---EVALUANDO Y ORDENANDO IMPORTANCIA DE DOCUMENTOS--- Step {step_count}/10")
    print(f"RESPECTO A LA PREGUNTA: {state['question']}")
    # question = state.get("augmented_question", state["question"])
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    # redirect = "Yes"
    found_relevant = False

    for i, d in enumerate(documents):
        print(f"---EVALUANDO DOCUMENTO {i+1}---")
        print(f"Document: {d.page_content}\n")  # Snippet del documento

        score = retrieval_grader_relevance.invoke(
            {"question": question, "document": d.page_content}
        )

        print(
            f"RESPUESTA DEL EVALUADOR: {score}\n"
        )  # Muestra todo el objeto con .binary_score

        if score.binary_score.lower() in ["sí", "si", "yes"]:
            print(f"EVALUACIÓN: DOCUMENTO {i+1} RELEVANTE\n")
            filtered_docs.append(d)
            found_relevant = True
        else:
            print(f"EVALUACIÓN: DOCUMENTO {i+1} NO RELEVANTE\n")

    state["documents"] = filtered_docs
    state["redirect"] = "No" if found_relevant else "Yes"
    state["step_count"] = step_count
    if found_relevant:
        state["__ui_stage"] = "🧪 EVALUANDO DOCUMENTOS — ✅ Hay contexto relevante"
    else:
        state["__ui_stage"] = (
            "🧪 EVALUANDO DOCUMENTOS — ❌ Sin contexto suficiente, se irá a web"
        )

    return state


# WEB SEARCH
#  NODO OMITIDO EN VERSION FINAL, YA QUE SE HA DECIDIDO USAR LA REDIRECCION DE LA CONVERSACION, Y NO USAR LA BÚSQUEDA WEB, 
# PARA EVITAR RESPUESTAS INCOHERENTES O FUERA DE CONTEXTO


# def web_search(state: GraphState) -> GraphState:
#     step_count = state.get("step_count", 0) + 1
#     print(f"---REALIZANDO BÚSQUEDA WEB--- Step {step_count}/10")
#     question = state.get("augmented_question", state["question"])
#     documents = state["documents"]

#     docs = web_search_tool.invoke({"query": question})
#     try:
#         web_results = "\n".join(
#             [
#                 f"{d['url']}\n{d['content']}" if isinstance(d, dict) else str(d)
#                 for d in docs
#             ]
#         )
#     except Exception as e:
#         web_results = "\n".join(str(d) for d in docs)
#         print("⚠️ Error formateando resultados de web search:", e)
#     web_results = Document(page_content=web_results)

#     documents.append(web_results)
#     state["documents"] = documents
#     state["step_count"] = step_count
#     return state


def redirect(state: GraphState) -> GraphState:
    state["__ui_stage"] = (
        f"🌐 BÚSQUEDA WEB / RESPUESTA ALTERNATIVA — Step {state.get('step_count',0)+1}/10"
    )

    step_count = state.get("step_count", 0) + 1
    print(
        f"---PREGUNTA FUERA DE ALCANCE, GENERANDO RESPUESTA PERSONALIZADA--- Step {step_count}/10"
    )

    question = state.get("augmented_question", state["question"])
    history = state.get("chat_history", [])

    history_text = "\n".join([f"{role}: {msg}" for role, msg in history[-6:]])

    # ✅ Determinar el modelo seleccionado por el usuario
    modelo = state.get("modelo_llm", "deepseek")

    # muestra el nombre del objeto llm
    print("OBJETO LLM SELECCIONADO:", llm.__class__.__name__)

    print("SE GENERA RESPUESTA CON EL MODELO (redirect):", modelo)
    llm_actual = deepseek if modelo == "deepseek" else openai

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un agente, y estás recibiendo esta pregunta porque no se ha relacionado con tu propósito. Tu propósito es ayudar con temas de inversión, "
                "finanzas personales, economía, educación financiera, etc... Si recibes una pregunta fuera de ese ámbito, "
                "deberás sugerir con cierto humor inteligente que el usuario te pregunte algo más relacionado con tu especialidad."
                "También recibirás historial reciente de conversación, si existe, y podrás tenerlo en cuenta para enriquecer tu respuesta."
                "Comenta que esta pregunta no se ajusta al propósito del asistente, y pide por favor que para una correcta utilización del asistente, formule preguntas relacionadas con inversión, finanzas personales, economía, educación financiera, etc... "
                "Puedes usar el historial para intentar conectar la pregunta con esos temas, aunque la formulación actual no tenga relación directa. "
                "Si no es posible encontrar conexión, simplemente aclara que no puedes responder a esa pregunta y anima al usuario a hacer preguntas relacionadas con tu especialidad.",
            ),
            (
                "human",
                "Pregunta del usuario:\n{question}\n\nHistorial reciente (si hay):\n{history}",
            ),
        ]
    )

    filled_prompt = prompt.invoke({"question": question, "history": history_text})
    # respuesta = llm_actual.invoke(filled_prompt)
    respuesta = llm.invoke(filled_prompt)
    state["generation"] = respuesta.content.strip()
    state["documents"] = []
    state["step_count"] = step_count
    return state


# DECIDE IF GENERATE OR ENRICH
def decide_to_generate(state: GraphState) -> str:
    print(f"---SE DECIDE GENERAR RESPUESTA--- Step {state['step_count']}/10")
    if state["redirect"] == "Yes":
        print("Sin docs relevantes en vectorstore → intentando yfinance...")
        return "enrich"
    return "generate"


# GENERATE
def generate(state: GraphState) -> GraphState:
    state["__ui_stage"] = (
        f"✍️ GENERANDO RESPUESTA — Step {state.get('step_count',0)+1}/10"
    )

    step_count = state.get("step_count", 0) + 1
    print(f"---GENERANDO RESPUESTA--- Step {step_count}/10")

    question = state["question"]
    documents = state["documents"]
    history = state.get("chat_history", [])

    # Convertir el historial en texto
    history_text = "\n".join([f"{role}: {msg}" for role, msg in history[-6:]])
    full_context = "\n\n".join([doc.page_content for doc in documents])

    prompt_input = {
        "question": question,
        "context": f"Historial reciente:\n{history_text}\n\nContexto documental:\n{full_context}",
    }

    # generation = rag_chain.invoke(prompt_input)

    # modelo = state.get("modelo_llm", "deepseek")
    # llm_usado = deepseek if modelo == "deepseek" else openai
    # generation = rag_chain | llm_usado | StrOutputParser()
    # print("SE GENERA RESPUESTA CON EL MODELO: ", modelo)
    # state["generation"] = generation.invoke(prompt_input)

    modelo = state.get("modelo_llm", "deepseek")
    print("OBJETO LLM SELECCIONADO:", llm.__class__.__name__)
    print("SE GENERA RESPUESTA CON EL MODELO:", modelo)
    llm_actual = deepseek if modelo == "deepseek" else openai

    local_prompt = ChatPromptTemplate.from_template(
        """Eres un asistente para tareas de pregunta y respuesta. Usa las siguientes partes de contexto extraído para responder a la pregunta. 
    Es importante que, aunque te otorgo cierta libertad, intentes no salirte del contexto proporcionado.
    Ten en cuenta el historial de conversación reciente, si existe, y utilízalo para enriquecer tu respuesta, si fuera necesario.
    Recuerda, es importante que no hagas referencia a tu propia existencia como asistente, ni a la de los documentos que has utilizado para responder, nada de "se menciona..." o "aparece..." en relacion al contexto que estas utilizando.
    Por ejemplo, no digas "según el documento", "según el contexto " o "según la información que tengo", simplemente responde a la pregunta del usuario con ese contexto.
    Pregunta: {question}
    Contexto: {context}
    Respuesta:"""
    )

    # rag_chain = local_prompt | llm_actual | StrOutputParser()
    rag_chain = local_prompt | llm | StrOutputParser()

    generation_output = rag_chain.invoke(prompt_input)

    state["generation"] = generation_output

    # state["generation"] = generation
    state["step_count"] = step_count
    return state


# GRADE GENERATION
def grade_generation_v_documents_and_question(state: GraphState) -> str:
    state["__ui_stage"] = (
        f"🧠 COMPROBANDO RESPUESTA — Step {state.get('step_count',0)+1}/10"
    )

    step_count = state.get("step_count", 0) + 1
    print(f"---COMPROBANDO ALUCINACIONES--- Step {step_count}/10")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    history = state.get("chat_history", [])

    # Construir historial como texto
    history_text = "\n".join([f"{role}: {msg}" for role, msg in history[-6:]])
    print("---HISTORIAL RECIENTE PRESENTE PARA EVALUACIÓN---")
    print(history_text + "\n")

    # Pasar historial también al evaluador de alucinaciones si es útil
    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    print(
        f"RESULTADO DE COMPROBADOR DE ALUCINACIONES: {hallucination_score.binary_score.upper()}"
    )

    if hallucination_score.binary_score.lower() in ["sí", "si", "yes"]:
        state["__ui_stage"] = (
            f"🧠 COMPROBANDO RESPUESTA — Step {state.get('step_count',0)+1}/10"
        )

        print("---COMPROBANDO RELEVANCIA DE LA RESPUESTA---")
        print(
            "Evaluando si la respuesta aborda correctamente la pregunta del usuario..."
        )

        # Enriquecer pregunta con historial para la evaluación
        question_with_history = (
            f"Historial reciente:\n{history_text}\n\nPregunta actual:\n{question}"
        )

        answer_score = answer_grader.invoke(
            {"question": question_with_history, "generation": generation}
        )
        print("PREGUNTA DEL USUARIO:", question_with_history)
        print("RESPUESTA GENERADA:", generation)
        print(
            f"RESULTADO DEL COMPROBADOR DE RESPUESTA: {answer_score.binary_score.upper()}"
        )

        if answer_score.binary_score.lower() in ["sí", "si", "yes"]:
            state["__ui_stage"] = "✅ RESPUESTA ÚTIL"

            print("---DECISIÓN: LA RESPUESTA GENERADA ES ÚTIL---\n")
            # print("LA RESPUESTA GENERADA ES:\n")
            # print(generation)
            print("\n\nRESPUESTA GENERADA, ESPERANDO NUEVA PREGUNTA...\n")
            return "useful"
        else:
            state["__ui_stage"] = "↪️ RESPUESTA NO ÚTIL — se hará búsqueda web"

            print("---DECISIÓN: RESPUESTA NO ÚTIL, SE HARÁ BÚSQUEDA WEB---\n")
            return "not useful"
    else:
        state["__ui_stage"] = "🔁 RESPUESTA NO RESPALDADA — reintentando generación"

        print("---DECISIÓN: RESPUESTA NO RESPALDADA, SE INTENTA NUEVA GENERACIÓN---\n")
        return "not supported"


# MARKET DATA ENRICHMENT (tool calling con yfinance)
from market_tools import MARKET_TOOLS, TOOLS_BY_NAME
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def market_data_enrichment(state: GraphState) -> GraphState:
    """
    Nodo de enriquecimiento entre `grade_documents` (rama 'relevantes') y `generate`.

    Ejecuta un bucle agéntico (ReAct con tool calling): el LLM decide qué
    herramientas de yfinance invocar; al recibir los resultados, puede pedir
    nuevas llamadas (p.ej. tras `search_ticker` pedir `get_quote`,
    `get_historical`, `get_news`…). Cada resultado se añade como un `Document`
    con `metadata["source"] = "yfinance"` a `state["documents"]`, de modo que
    `generate` recibe el contexto RAG original + los datos de mercado.
    """
    state["__ui_stage"] = (
        f"🛠️ ENRIQUECIENDO CON DATOS DE MERCADO — Step {state.get('step_count',0)+1}/10"
    )
    step_count = state.get("step_count", 0) + 1
    print(f"---ENRIQUECIENDO CON DATOS DE MERCADO (YFINANCE)--- Step {step_count}/10")
    state["step_count"] = step_count

    question = state.get("augmented_question") or state["question"]

    system_prompt = (
        "Eres un agente financiero con acceso a herramientas de Yahoo Finance "
        "(yfinance). Tu objetivo es recoger datos ACTUALES de mercado que "
        "complementen la respuesta a la pregunta del usuario.\n\n"
        "Reglas:\n"
        "1) Si la pregunta es puramente teórica o de educación financiera "
        "(ej. 'qué es un ETF', 'cómo funciona el interés compuesto'), NO "
        "llames a ninguna herramienta.\n"
        "2) Si aparece un ticker, nombre de empresa, fondo/ETF, materia prima, "
        "divisa o criptomoneda, DEBES construir un contexto completo. Para "
        "cada símbolo relevante recoge:\n"
        "   • Cotización actual con `get_quote`.\n"
        "   • Evolución reciente con `get_historical` (period='1mo' o '6mo').\n"
        "   • Fundamentales con `get_fundamentals` (si es una acción) o "
        "`get_fund_info` (si es ETF/fondo).\n"
        "   • Noticias recientes con `get_news`.\n"
        "3) Si el usuario da un NOMBRE y no un ticker, primero llama a "
        "`search_ticker`. Cuando recibas los candidatos, ELIGE el más "
        "adecuado y en la siguiente ronda llama a las herramientas de datos "
        "con ese símbolo. No te detengas tras `search_ticker`.\n"
        "4) Puedes llamar a varias herramientas en paralelo en una misma "
        "ronda. No repitas la misma llamada con los mismos argumentos.\n"
        "5) Cuando ya tengas suficiente contexto, responde sin emitir más "
        "tool calls (tu respuesta textual se descarta; lo que importa son los "
        "datos recogidos).\n"
    )

    try:
        llm_with_tools = llm.bind_tools(MARKET_TOOLS)
    except Exception as e:
        print(f"[enrich] Error al bindear tools: {e}")
        return state

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    market_docs: List[Document] = []
    seen_calls: set = set()
    MAX_ITERATIONS = (
        3  # p.ej. ronda 1: search_ticker / ronda 2: datos / ronda 3: extras
    )

    for iteration in range(MAX_ITERATIONS):
        try:
            ai_response = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"[enrich] Error al invocar LLM con tools (iter {iteration}): {e}")
            break

        messages.append(ai_response)
        tool_calls = getattr(ai_response, "tool_calls", None) or []
        if not tool_calls:
            print(f"[enrich] LLM no pide más herramientas (iter {iteration}).")
            break

        print(
            f"[enrich] iter {iteration}: {len(tool_calls)} tool call(s) → "
            f"{[tc.get('name') for tc in tool_calls]}"
        )

        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            call_id = tc.get("id") or f"{name}-{iteration}"
            try:
                args_key = json.dumps(args, sort_keys=True, default=str)
            except Exception:
                args_key = str(args)
            dedup_key = (name, args_key)

            fn = TOOLS_BY_NAME.get(name)
            if fn is None:
                result = f"Tool desconocida: {name}"
            elif dedup_key in seen_calls:
                result = f"(Llamada duplicada a {name} con {args}, omitida)"
            else:
                try:
                    result = fn.invoke(args)
                except Exception as e:
                    result = f"Error ejecutando {name}({args}): {e}"
                seen_calls.add(dedup_key)
                header = f"[yfinance · {name}({args})]"
                market_docs.append(
                    Document(
                        page_content=f"{header}\n{result}",
                        metadata={
                            "source": "yfinance",
                            "tool": name,
                            "args": args,
                            "summary": f"Datos de mercado ({name}): {list(args.values())}",
                        },
                    )
                )
            # Devolver al LLM el ToolMessage para que pueda planificar la siguiente ronda.
            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    existing = state.get("documents") or []
    state["documents"] = list(existing) + market_docs
    print(f"[enrich] Añadidos {len(market_docs)} documento(s) de yfinance al contexto.")
    return state


# BUILD WORKFLOW
workflow = StateGraph(GraphState)
workflow.add_node("augment", augment)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("enrich", market_data_enrichment)
workflow.add_node("generate", generate)
workflow.add_node("redirect", redirect)

workflow.set_entry_point("augment")


# Función envoltorio para que el nodo devuelva un dict
def route_node(state: GraphState) -> GraphState:
    return state  # solo pasa el estado sin modificar


workflow.add_node("route_question", route_node)
workflow.add_edge("augment", "route_question")

workflow.add_conditional_edges(
    "route_question",
    route_question,
    {
        "redirect": "redirect",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "enrich")
workflow.add_edge("enrich", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "not useful": "redirect",
        "useful": END,
    },
)

# COMPILE
app = workflow.compile()
