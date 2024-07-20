import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Loading environment variables from .env file
load_dotenv()

# Function to initialize conversation chain with GROQ language model
groq_api_key = os.environ['GROQ_API_KEY']

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-70b-8192",
    temperature=2
)

# Path to save/load ChromaDB
CHROMA_DB_PATH = "Finaldb"

@cl.on_chat_start
async def on_chat_start():
    # List of hardcoded PDF file paths
    pdf_file_paths = [ "citizenship.pdf","1.pdf","2.pdf",  "3.pdf",  "4.pdf",  "5.pdf",  "6.pdf",  "7.pdf",  "8.pdf",  "9.pdf",  
                      "10.pdf",  "11.pdf",  "12.pdf",  "13.pdf",  "14.pdf",  "15.pdf",  "16.pdf",  "17.pdf",  "18.pdf",  "19.pdf",  "20.pdf",  
                      "21.pdf",  "22.pdf",  "23.pdf",  "24.pdf",  "25.pdf",  "26.pdf", "Civil-code.pdf","con.pdf","mulki.pdf"               
    ]

    # Check if ChromaDB already exists
    if os.path.exists(CHROMA_DB_PATH):
        # Load the existing ChromaDB
        docsearch = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
    else:
        # Process each PDF file
        texts = [] #empty dictionary is created 
        metadatas = []
        for file_path in pdf_file_paths:
            print(file_path)  # Print the file path for debugging

            # Read the PDF file
            with open(file_path, "rb") as file:
                pdf = PyPDF2.PdfReader(file)
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text()

                # Split the text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
                file_texts = text_splitter.split_text(pdf_text)
                texts.extend(file_texts) #above empty  dictionary is appended with the splitted texts

                # Create metadata for each chunk
                file_metadatas = [{"source": f"{i}-{os.path.basename(file_path)}"} for i in range(len(file_texts))]
                metadatas.extend(file_metadatas)

        # Create a Chroma vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas, persist_directory=CHROMA_DB_PATH
        )

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Inform the user that processing has ended. You can now chat.
    msg = cl.Message(content=f"Processing {len(pdf_file_paths)} files done. You can now ask questions!")
    await msg.send()

    # Store the chain in the user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")
    # Callbacks happen asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with the user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    # Return results
    await cl.Message(content=answer, elements=text_elements).send()