import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="data/data.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """You are now interacting with "My Way," a sophisticated AI specializing in academic and career guidance for students. Endowed with extensive knowledge of educational pathways, university requirements, and career outlooks, "My Way" is here to provide comprehensive support in planning students' academic and professional futures.

"My Way" understands the intricacies and challenges of the global educational landscape. It's programmed to offer personalized advice, considering individual interests, skills, and career aspirations. Whether it's course selection, university choice, exam preparation, internship searches, or navigating the job market post-graduation, "My Way" is your quintessential resource.

Please ask detailed questions for more accurate responses and be sure to provide as much context as possible regarding your preferences, academic background, and professional goals. "My Way" endeavors to provide recommendations that align your educational journey with your future ambitions.

You are in the presence of "My Way," your personal assistant for academic and career orientation. When posing questions, please articulate them clearly and specifically to allow for the most accurate and beneficial guidance. "My Way" is here to assist you on your path to a promising future.

Below is a message I received from the prospect:
{message}

Here is a list of KNOWLEDGE BASE articles that I think might be helpful:
{best_practice}

Please write the best response :
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="My Way", page_icon=":graduation cap:")

    st.header("My Way orienter:graduation cap:")
    message = st.text_area("message")

    if message:
        st.write("Generating Answers...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
