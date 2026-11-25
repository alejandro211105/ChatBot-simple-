"""
Chatbot con memoria de conversacion en sesion actual usando Qwen2.5-0.5B-Instruct y Gradio.
No persiste datos entre sesiones.
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuracion del modelo
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 512
SYSTEM_PROMPT = (
    "Eres un asistente virtual util y amigable. "
    "Responde de forma clara y concisa. "
    "Si te hablan en espanol, responde en espanol."
)


def load_model():
    """Carga modelo y tokenizador desde HuggingFace."""
    print(f"[...] Cargando modelo '{MODEL_NAME}'...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    print("[OK] Modelo cargado.")
    return model, tokenizer


def respond(message, history):
    """Genera respuesta usando historial de la sesion actual.

    Args:
        message: Texto del usuario.
        history: Lista de [user, assistant] de la sesion actual.

    Returns:
        Respuesta generada.
    """
    if not message or not message.strip():
        return "Por favor, escribe una pregunta."

    # Construir mensajes con historial
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Preparar entrada con plantilla de chat
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenizar
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # Generar respuesta
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # Decodificar solo tokens nuevos
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response


if __name__ == "__main__":
    model, tokenizer = load_model()

    interface = gr.ChatInterface(
        fn=respond,
        title="ChatBot con Memoria",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Escribe tu mensaje aqui...", scale=7),
    )

    print("\n[>>] Iniciando servidor...")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
