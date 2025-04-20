import streamlit as st
import tensorflow as tf
import Chatbot_class
from preprocess import clean_text
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Page setup
st.set_page_config(page_title="Bilingual Chatbot", layout="centered")
st.title("🤖 Bilingual ChatBot")

# --- MODE SELECTION ---
mode = st.sidebar.selectbox(
    "🧠 Choose Chatbot Mode:",
    ["Seq 2 Seq Model", "Hugging Face Model"]
)

# --- CACHE LOADING FUNCTIONS ---


@st.cache_resource
def load_custom_chatbot():
    with open('./processed_data/inp_lang.json', 'r') as f:
        inp_lang = tokenizer_from_json(json.load(f))
    with open('./processed_data/targ_lang.json', 'r') as f:
        targ_lang = tokenizer_from_json(json.load(f))

    embedding_dim = 128
    units = 256
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    max_len = 15

    encoder = Chatbot_class.create_encoder(
        vocab_inp_size, embedding_dim, units, max_len)
    encoder.load_weights('trained_model/encoder_weights.h5')

    decoder = Chatbot_class.create_decoder(
        vocab_tar_size, embedding_dim, units, units, max_len)
    decoder.load_weights('trained_model/decoder_weights.h5')

    return inp_lang, targ_lang, encoder, decoder, vocab_tar_size, max_len


@st.cache_resource
def load_french_chatbot():
    # Can be replaced with better French models
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# --- EVALUATION FUNCTIONS ---


def evaluate_custom(sentence, samp_type, inp_lang, targ_lang, encoder, decoder, vocab_tar_size, max_len):
    sentence = clean_text(sentence)
    inputs = [inp_lang.word_index.get(
        w, inp_lang.word_index['<unk>']) for w in sentence.split()]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_len, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    enc_output, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for _ in range(max_len):
        predictions, dec_hidden = decoder([enc_output, dec_hidden, dec_input])
        prediction_probs = predictions[0].numpy()

        if samp_type == 1:
            predicted_id = tf.argmax(predictions[0]).numpy()
        elif samp_type == 2:
            predicted_id = np.random.choice(vocab_tar_size, p=prediction_probs)
        elif samp_type == 3:
            _, top_indices = tf.math.top_k(predictions[0], k=3)
            predicted_id = np.random.choice(top_indices.numpy())

        if predicted_id != 0:
            predicted_word = targ_lang.index_word.get(predicted_id, '')
            if predicted_word == '<end>':
                return result.strip()
            result += predicted_word + ' '

        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()


def evaluate_french(message, history):
    tokenizer, model = load_french_chatbot()

    new_input_ids = tokenizer.encode(
        message + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat(
        history + [new_input_ids], dim=-1) if history else new_input_ids

    output_ids = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, history + [new_input_ids]


# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "hf_history" not in st.session_state:
    st.session_state.hf_history = []

# --- USER INPUT ---
with st.form(key="chat_form"):
    user_input = st.text_input(
        "💬 Your message", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

# --- PROCESS INPUT ---
if submitted and user_input:
    if mode == "Seq 2 Seq Model":
        # Load models
        inp_lang, targ_lang, encoder, decoder, vocab_tar_size, max_len = load_custom_chatbot()
        # Sampling strategy
        sampling = st.sidebar.radio("🎯 Sampling Strategy", [
                                    "Greedy", "Probabilistic", "Top-3"])
        sampling_map = {"Greedy": 1, "Probabilistic": 2, "Top-3": 3}
        samp_type = sampling_map[sampling]

        # Get response
        response = evaluate_custom(
            user_input, samp_type, inp_lang, targ_lang, encoder, decoder, vocab_tar_size, max_len)
        st.session_state.chat_history.append(("🧑", user_input))
        st.session_state.chat_history.append(("🤖", response))

    elif mode == "Hugging Face Model":
        response, updated_history = evaluate_french(
            user_input, st.session_state.hf_history)
        st.session_state.hf_history = updated_history
        st.session_state.chat_history.append(("🧑", user_input))
        st.session_state.chat_history.append(("🤖", response))

# --- DISPLAY CHAT HISTORY ---
st.markdown("---")
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}**: {message}")
