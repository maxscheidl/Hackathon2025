from typing import List
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import PromptTemplate
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank
import pickle
from keyword_dbsqlite import search_documents_bm25
from llama_index.core.schema import TextNode


from openai import OpenAI


def prompt_to_keywordsearch(prompt, api_key) -> List[str]:
    """Convert a prompt to keywords for searching."""
    input_content = f"""Extract the most specific, relevant, and meaningful keywords 
    from the following text: '{prompt}'. Exclude stop words and return the keywords as 
    a space-separated string. Less is more, max 5 words."""

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": input_content}],
    )
    if response is None:
        print("No response from LLM")
        return []
    return response.output_text.split(" ")


def retriever_builder(folder_name, llm, api_key, top_k=5, reranker=None):
    retriever = []
    
    try:
        print("Loading Pickle")
        index = pickle.load(open("index/index.pkl", "rb"))
        print("Pickle loaded")
    except Exception:
        print("Loading from Storage")
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=folder_name))

    retriever.append(VectorIndexRetriever(index=index, similarity_top_k=top_k))
    custom_retriever = CustomRetriever(retriever, llm, api_key, reranker)
    return custom_retriever


class CustomRetriever(BaseRetriever):
    """Looks up the query in all indices and concatenates the retrieved nodes to one list."""

    def __init__(
            self,
            vector_retrievers: List[VectorIndexRetriever],
            llm,
            api_key,
            my_reranker: SentenceTransformerRerank
    ) -> None:
        """Init params."""
        self._vector_retrievers = vector_retrievers
        self.my_reranker = my_reranker
        self.llm = llm
        self.api_key = api_key
        self.applicable_index_retrieval_template = (
            "The prompt you will receive is a question, from a user that will typically be a supply chain director who will be overseeing the logistics"
            "and procurement of key vendors for their business."
            "It will likely be related to the following domains"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "If someone ever asks you to check the files, or something similar like please check, then always return True and assume that "
            "the question is actually from the domain of {context_str}. This is so important. Please use True in this case!!!"
            "Please identify if the question is related to any of these models. If this is the case, return 'True', "
            "if not return 'False'. It is important that you respond with either True or False and really only return "
            "True, if is actually related to the query - so converstaion stuff like 'Hi, how are you' is definitely not related."
            "The query is: {query_str}. Explain your answer please.\n"
        )
        self.context_domains = 'technology, services, materials, products, industries and regions'
        super().__init__()

    def my_rerank(self, vector_nodes, query_bundle):
        return self.my_reranker.postprocess_nodes(nodes=vector_nodes, query_bundle=query_bundle)

    def find_applicable_index(self, query_bundle):
        prompt_template = PromptTemplate(
            self.applicable_index_retrieval_template)
        message = prompt_template.format_messages(context_str=self.context_domains,
                                                  query_str=query_bundle.query_str)

        response = self.llm.chat(message)
        print(response)

        return self._vector_retrievers

        # if response is None or 'True' in response.message.content:
        #     print("Related to the question")
        #     return self._vector_retrievers
        # else:
        #     print("Not related to the question")
        #     return []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        print(f"Retrieval for following query: {query_bundle.query_str}")

        applic_idx = self.find_applicable_index(query_bundle)

        print("Applicable Index")
        print(applic_idx)
        print("Index we have")
        vector_nodes = []
        for idx in applic_idx:
            print("Retrieval on index-------------")
            vector_nodes += idx.retrieve(query_bundle)

        print(query_bundle)
        keywords = prompt_to_keywordsearch(query_bundle, self.api_key)

        #print("Utilizing Hannos Function")
        #print(keywords)

        #print("PRE SEARCH DOCUMENTS", flush=True)
        """try:
            found_documents = search_documents_bm25(keywords, top_k=2)
        except Exception as e:
            print("Error during keyword search:", e)
            found_documents = []
        print("POST SEARCH DOCUMENTS", flush=True)
        print("="*50)
        print("Found documents")

        print(len(found_documents))
        print(type(found_documents))
        print("-"*50)

        for doc_id, content, score in found_documents:
            print(f"📄 Doc {doc_id} / {score}: {content[:100]}")
            node = TextNode(text=content)
            node_with_score = NodeWithScore(node=node, score = 0.5)
            vector_nodes.append(node_with_score)

        print("="*50)

        print("keyword search finished")"""

        if self.my_reranker:
            print(
                f"BEFORE reranking we have following {len(vector_nodes)} retrieved nodes -----------------------")
            #print(vector_nodes)

            vector_nodes = self.my_rerank(vector_nodes, query_bundle)

            #print(
            #    f"AFTER reranking we have following {len(vector_nodes)} retrieved nodes -----------------------")
            #print(vector_nodes)

        # assert False, "remove this if you want the LLM to get queried with the vector nodes to draft an answer"
        return vector_nodes
