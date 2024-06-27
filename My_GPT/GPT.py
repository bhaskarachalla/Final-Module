# Enhanced Q&A with Sentiment Analysis
import streamlit as st
import streamlit_option_menu
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, BartForConditionalGeneration, BartTokenizer, AutoModelForQuestionAnswering, pipeline
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(page_title="AI Model Expo", page_icon="🧠", layout="wide")

# Title and header
st.markdown("<h1 style='text-align: center; color: blue;'>---->GPT<----</h1>", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.title('WHAT CAN I DO ?')
    selected = streamlit_option_menu.option_menu(
        None,
        ['PREDICTING WORDS', "GENERATING TEXT", "SUMMARIZING", "CHATTING", "IMAGE GENERATION", "QUESTION ANSWERING", "SENTIMENT ANALYSIS"],
        icons=['pencil', 'file-text', 'file-earmark-text', 'chat', 'image', 'question-circle', 'emoji-smile'],
        orientation='vertical',
        default_index=0
    )

@st.cache_resource
def load_models(device):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    gpt2_model.eval()

    bart_model_name = "facebook/bart-large-cnn"
    bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)
    bart_model.eval()

    dilo_model_name = "microsoft/DialoGPT-medium"
    dilo_tokenizer = AutoTokenizer.from_pretrained(dilo_model_name, padding_side='left')
    dilo_model = AutoModelForCausalLM.from_pretrained(dilo_model_name).to(device)
    dilo_model.eval()

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        revision="fp16" if torch.cuda.is_available() else "main"
    ).to(device)

    qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(device)

    sentiment_analysis = pipeline("sentiment-analysis")

    return gpt2_tokenizer, gpt2_model, bart_tokenizer, bart_model, dilo_tokenizer, dilo_model, pipe, qa_tokenizer, qa_model, sentiment_analysis

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and tokenizers
gpt2_tokenizer, gpt2_model, bart_tokenizer, bart_model, dilo_tokenizer, dilo_model, pipe, qa_tokenizer, qa_model, sentiment_analysis = load_models(device)

# Functions for NLP tasks
def predict_next_word(prompt, model, tokenizer, top_k=1):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]
    top_k_tokens = torch.topk(next_token_logits, top_k).indices[0].tolist()
    predicted_tokens = [tokenizer.decode([token]) for token in top_k_tokens]
    return predicted_tokens

def generate_text(prompt, model, tokenizer, max_length=36):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize(text, model, tokenizer, max_length=180, min_length=36, do_sample=False):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def chat_with_model(prompt, chat_history_ids, model, tokenizer):
    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt').to(device)
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def generate_image(prompt, pipe):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

def answer_question(question, context, model, tokenizer):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

def analyze_sentiment(texts, sentiment_analysis):
    results = sentiment_analysis(texts)
    return results

# Streamlit interface
st.title("AI Model Explorer")

if selected == 'PREDICTING WORDS':
    st.header("Word Predictor")
    prompt = st.text_input("Enter a prompt for GPT-2:", "--Enter text here--")
    if st.button("Predict Next Word"):
        predictions = predict_next_word(prompt, gpt2_model, gpt2_tokenizer)
        st.write("Next word predictions:", predictions)

if selected == 'GENERATING TEXT':
    st.header("Text Generator")
    prompt = st.text_input("Enter a prompt for GPT-2:", "--Enter text here--")
    if st.button("Generate Text"):
        generated_text = generate_text(prompt, gpt2_model, gpt2_tokenizer)
        st.write("Generated text:", generated_text)

if selected == 'SUMMARIZING':
    st.header("Text Summarizer")
    text = st.text_area("Enter text to summarize:", "--Enter text to summarize--")
    if st.button("Summarize Text"):
        summary = summarize(text, bart_model, bart_tokenizer)
        st.write("Summary:", summary)

if selected == 'CHATTING':
    st.header("Chat with DialoGPT")
    chat_prompt = st.text_input("Enter your message to chat with DialoGPT:", "--Enter message here--")
    if 'chat_history_ids' not in st.session_state:
        st.session_state['chat_history_ids'] = None
    if 'response' not in st.session_state:
        st.session_state['response'] = ""

    if st.button("Send Message"):
        response, chat_history_ids = chat_with_model(chat_prompt, st.session_state['chat_history_ids'], dilo_model, dilo_tokenizer)
        st.session_state['chat_history_ids'] = chat_history_ids
        st.session_state['response'] = response

    st.write(st.session_state['response'])

if selected == 'IMAGE GENERATION':
    st.header("Image Generator")
    image_prompt = st.text_input("Enter a prompt to generate an image:", "--Enter image prompt here--")
    if st.button("Generate Image"):
        image = generate_image(image_prompt, pipe)
        st.image(image, caption="Generated Image")

if selected == 'QUESTION ANSWERING':
    st.header("Question Answering")
    context = st.text_area("Enter the context:", "--Enter context here--")
    question = st.text_input("Enter your question:", "--Enter question here--")
    if st.button("Get Answer"):
        answer = answer_question(question, context, qa_model, qa_tokenizer)
        st.write("Answer:", answer)

if selected == 'SENTIMENT ANALYSIS':
    st.header("Sentiment Analysis")
    texts = st.text_area("Enter texts for sentiment analysis (separated by newlines):", "--Enter texts here--").split('\n')
    if st.button("Analyze Sentiment"):
        results = analyze_sentiment(texts, sentiment_analysis)
        for text, result in zip(texts, results):
            st.write(f"Text: {text}")
            st.write(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")
