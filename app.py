import streamlit as st
import helper
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.embeddings.openai import OpenAIEmbedding

# imports for CustomRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

# chat engine
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.tools.query_engine import QueryEngineTool
# -------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Gieni Lookup Bot",
    page_icon="🔎"
)

st.title("🔎 Gieni Lookup Bot")

st.sidebar.title("🚀 Model Parameters")
st.sidebar.markdown("Configure the settings for the chatbot below.")

topk_reranker = st.sidebar.number_input(
    "Top k results",
    min_value=1,
    max_value=20,
    value=5
)

max_tokens = st.sidebar.number_input(
    "Max Tokens",
    min_value=1,
    max_value=2000,
    value=100
)

# API Key input field with submit button
api_key = st.sidebar.text_input("Enter API Key:", type="password")
# azure_endpoint = st.sidebar.text_input(
#     "Enter Azure Endpoint:", type="password")
if st.sidebar.button("Submit API Key & start with this configuration"):
    if api_key:  # Check if API key is not empty
        st.sidebar.success("Credentials submitted successfully!")
        st.session_state["api_key"] = api_key
    else:
        st.sidebar.error(
            "Please enter a valid API Key & endpoint before submitting.")

#if st.sidebar.button("Clear chat history"):
#    st.session_state["messages"] = []

 #   agent = AgentRunner.from_llm(
 #       # this is exactly what happens internally if you call the .as_chat_engine()
 #       tools=[st.session_state["query_engine_tool"]],
 #       llm=st.session_state["llm"],
 #       system_prompt="Whenever you get any nodes or results from you tools, always try to combine them with the query you get. This is super important."
 #   )

 #   st.session_state["chat_engine"] = agent



if "api_key" in st.session_state:
    # Initialize chat engine & lookup dict
    if "chat_engine" not in st.session_state:
        #api_version = '2024-08-01-preview'
        model = 'gpt-4o-mini'

        llm = OpenAI(
            model=model,
            engine=model,
            max_tokens=max_tokens,
            api_key=api_key,
            temperature=0
        )

        # embed_model = OpenAIEmbedding(embed_batch_size=10, api_key=api_key)
        # embed_model = HuggingFaceEmbedding(model_name="ibm-granite/granite-embedding-125m-english")
        embed_model = OpenAIEmbedding(
            embed_batch_size=10,
            api_key=api_key,
            model="text-embedding-3-small"
        )


        Settings.llm = llm
        Settings.embed_model = embed_model

        reranker = LLMRerank(choice_batch_size=5, top_n=topk_reranker)
        #reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=topk_reranker)


#         indices_topk = {("briefings", 3), ("background", 3),
#                         ("CCC", 3), ("FCM", 3), ("FCTM", 3), ("FCOM", 5)}

        index_folder_name = "index/VectorDB"
        retriever_query_engine = RetrieverQueryEngine(
            retriever=helper.retriever_builder(index_folder_name, llm, api_key, 10, reranker),
            response_synthesizer=get_response_synthesizer()
        )
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=retriever_query_engine)

        st.session_state["query_engine_tool"] = query_engine_tool
        st.session_state["llm"] = llm

        agent = AgentRunner.from_llm(
            # this is exactly what happens internally if you call the .as_chat_engine()
            tools=[query_engine_tool],
            llm=llm,
            system_prompt="Whenever you get any nodes or results from you tools, always try to combine them with the query you get. This is super important."
        )

        st.session_state["chat_engine"] = agent

    #     if "lookup_dict" not in st.session_state:
#         st.session_state.lookup_dict = myXML.get_id_lookup_dict_for_final_title_print(
#             "./_MANUALS")

#     # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
#     # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

#     # Accept user input
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chat = st.session_state.chat_engine.chat(prompt)
            print_text = chat.response

            print("This is the output:")
            print(chat)
            # this is the case where we enter the retrievers but the query is unrelated
            if len(chat.sources) > 0 and len(chat.source_nodes) == 0:
                print_text = (
                    "I am unable to find any relevant information on this topic. Are you sure the query is related "
                    "to the manual? Please try again.")

            # data = {"Document": [], "Title": []}  # keep the order
            # seen_pairs = set()
            # for node in chat.source_nodes:
            #     document = st.session_state.lookup_dict[node.node.id_][1]
            #     title = st.session_state.lookup_dict[node.node.id_][0]
            #     if (document, title) not in seen_pairs:
            #         data["Document"].append(document)
            #         data["Title"].append(title)
            # if len(data["Title"]):
            #     print_text += "\n\n**Sources**"
            #     print_text += f"\n{myXML.dict_to_markdown_table(data)}"

            st.markdown(print_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": print_text})
else:
    st.warning("Please enter your API key to start the chatbot.")