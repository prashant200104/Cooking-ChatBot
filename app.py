# Import necessary libraries
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import databutton as db
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from brain import get_index_for_text
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set it in the environment variables.")
client = OpenAI(api_key=api_key)

# Set the title for the Streamlit app
st.title("Cooking Chatbot")

# Get the current prompt from the session state or set a default value
if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]
prompt = st.session_state["prompt"]

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# New input boxes for recipe, spice preference, and special ingredient
recipe_input = st.text_area("Please input recipe here , ここにレシピを入力してください：")
spice_input = st.text_input("Do you have any additional preferences , 追加の希望事項はありますか？")
ingredient_input = st.text_input("Any special ingredient you want to add , 追加したい特別な材料はありますか？")

# Merge the inputs
merged_input = f"Recipe: {recipe_input}\n , Spice Preference: {spice_input}\n , Special Ingredient: {ingredient_input}"

# Display the merged input
st.write("Merged Input:")
st.write(merged_input)

# Initialize vectordb using get_index_for_text
text_inputs = [recipe_input, spice_input, ingredient_input]  # Assuming these are your text inputs
vectordb = get_index_for_text(text_inputs, api_key)

# Now, set it in the session state
st.session_state["vectordb"] = vectordb

# Define the template for the chatbot prompt
prompt_template = """
    you are a helpful assistant which tells me about recipes, make changes in it according to me demand, and please print complete recipe in english and japanese
"""

# Handle the user's merged input
if st.button("Submit"):
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("You need to provide a text input")
            st.stop()

    # Get merged input from user
    merged_input = f"Recipe: {recipe_input}\n , Spice Preference: {spice_input}\n , Special Ingredient: {ingredient_input}"

    # Search the vectordb for similar content to the user's merged input
    search_results = vectordb.similarity_search(merged_input, k=5)
    similar_texts = [result.page_content for result in search_results]

    # Update the prompt with the similar texts
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(text_extract="\n".join(similar_texts)),
    }

    # Add the user's merged input to the prompt and display it
    prompt.append({"role": "user", "content": merged_input})
    with st.chat_message("user"):
        st.write(merged_input)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call ChatGPT with streaming and display the response as it comes
    response = []
    try:
        for chunk in client.chat.completions.create(
            model="gpt-3.5-turbo", messages=prompt, stream=True, temperature=0.6):
            text = chunk.choices[0].delta.content

            if text is not None:
                response.append(text)
                result = "".join(response).strip()
                botmsg.write(result)
    except Exception as e:
        st.error(f"Error during API call: {e}")

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

