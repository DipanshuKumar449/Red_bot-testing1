
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import openai
import os
import glob
import streamlit as st
import csv
from datetime import datetime
from PIL import Image
import logging

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure logging format and level
logging.basicConfig(
    filename='red_bot.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

st.set_page_config(page_title="RED_BOT", page_icon="/Users/dipanshukumar/Desktop/RED_BOT/red_team.png")


def main():
    '''
    Main function:
    This function does the following:
    1. provides UI using Streamlit for RedBot
    2. does splitting of the text using Character Text Splitter
    3. performs vector embeddings using OpenAI Embeddings and does similarity search
    between questions and the documents

    Inputs: User Query, Video Transcript Documents
    Outputs : Vector embeddings from documents, Response from GPT 3.5 Turbo model

    '''
    # Set app title and information button
    col1, col2 = st.columns([9.65, 0.35])
    col1.title("Doubt Resolution Chatbot")
    with col2:
        st.markdown(
            """
            <style>
            .info-button {
                position: relative;
                display: inline-block;
            }
            .info-content {
                position: absolute;
                top: 30px;
                right: -10px;
                width: 400px;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                display: none;
                z-index: 1;
            }
            .info-button:hover .info-content {
                display: block;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="info-button">ℹ️<span class="info-content">'
            'This chatbot uses a collection of conversations to find answers to your questions. Its purpose is to clear any doubts you have and give you the information you need based on the conversations it has learned from.'
            '<br><br>'
            'Here\'s how it works: the chatbot looks for similar content in the conversations and picks out the most relevant information to answer your question. It does this by using special techniques from OpenAI that help it understand the meaning and context of the text.'
            '<br><br>'
            'The chatbot has a question answering system that processes your question and finds the best answer from the conversations it has stored. It uses powerful language models and searches for similar text to make sure its responses are accurate and informative.'
            '<br><br>'
            '**Capabilities:**'
            '<ul>'
            '<li>Answers questions using the conversations it has learned from</li>'
            '<li>Searches for relevant information within the conversations</li>'
            '<li>Understands the meaning of the text using special techniques</li>'
            '<li>Keeps track of the conversation history for a smooth chat experience</li>'
            '</ul>'
            ''
            '</span></div>',
            unsafe_allow_html=True
        )

    # Display the logo image
    img = Image.open("/Users/dipanshukumar/Desktop/RED_BOT/472224_custom_site_themes_id_9D72vFZSIuIWH6zEYQg1_REDTEAM_ICON___LOGOTYPE_STACKED_RED.jpg")
    img_resized = img.resize((200, 200))  # Resize the image to desired dimensions
    st.sidebar.image(img_resized, use_column_width=True)
    

    # Initialize conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Initialize send button state
    if 'send_button_pressed' not in st.session_state:
        st.session_state.send_button_pressed = False

    # User input: Question
    question = st.text_input("User Input", key="user_input", value="", help="Enter your message...")

    if st.button("Send") or st.session_state.send_button_pressed:
            st.session_state.send_button_pressed = True
            if question:
                try:
                    # Load transcripts
                    logging.info("Loading transcripts...")
                    transcripts_lst = glob.glob("/Users/dipanshukumar/Documents/transcript/Transcripts/clean_transcripts.txt")
                    all_contents = ''
                    for i in transcripts_lst:
                        with open(i, 'r') as file:
                            all_contents += file.read()
                    logging.info("Transcripts loaded successfully.")

                    # Split text into chunks
                    logging.info("Splitting text into chunks...")
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1200,
                        chunk_overlap=200,
                        length_function=len,
                    )
                    texts = text_splitter.split_text(all_contents)
                    logging.info("Text split into chunks successfully.")

                    # Download embeddings from OpenAI
                    logging.info("Downloading embeddings from OpenAI...")
                    embeddings = OpenAIEmbeddings()
                    document_search = FAISS.from_texts(texts, embeddings)
                    logging.info("Embeddings downloaded successfully.")

                    # Load question answering chain
                    logging.info("Loading question answering chain...")
                    chain = load_qa_chain(OpenAI(), chain_type="stuff")
                    logging.info("Question answering chain loaded successfully.")

                    # Process the question and get the model's reply
                    logging.info("Processing the question...")
                    docs = document_search.similarity_search(question)
                    reply = chain.run(input_documents=docs, question=question)
                    logging.info("Question processed successfully.")

                    # Store conversation history
                    st.session_state.conversation.append((question, reply))

                    # Save conversation in a CSV file
                    save_conversation(question, reply)

                    # Display conversation history
                    logging.info("Displaying conversation history...")
                    display_conversation()
                    logging.info("Conversation history displayed.")
                except Exception as e:
                    logging.error("An error occurred while processing the request.")
                    logging.error(e)
                    st.error("An error occurred while processing the request. Please try again.")
            else:
                st.session_state.send_button_pressed = False

def save_conversation(question, answer):
    '''
    The save_conversation function is used to write the conversation between the user and bot
    to a text file.

    Input: question, answer
    Output: conversation.csv
    '''
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headers = ["Time", "Question Asked", "Answer"]

    with open('conversation.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if the file is empty
        if file.tell() == 0:
            writer.writerow(headers)
        writer.writerow([timestamp, question, answer])

def display_message(text, sender):
    '''
    The display_message function is created to display the user conversation
    and provides emojis for the same.
    '''
    if sender == "user":
        st.write(":man: " + text)
    else:
        st.write(":robot_face: " + text)


def display_conversation():
    '''
    The display_conversation function is created to display the user conversation
    between user and bot.
    '''
    st.subheader("Conversation History")
    for question, reply in st.session_state.conversation:
        display_message(question, "user")
        display_message(reply, "bot")

# Run the app
if __name__ == "__main__":
    main()