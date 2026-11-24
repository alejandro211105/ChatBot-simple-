"""
Chatbot simple usando Qwen2.5-0.5B-Instruct y Gradio.
Interfaz estilo chat sin memoria de conversacion.
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuracion del modelo ---
NOMBRE_MODELO = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS_RESPUESTA = 512


def cargar_modelo():
    """Carga el modelo y el tokenizador desde HuggingFace.
    La primera ejecucion descargara los pesos (~1 GB).
    """
    print(f"[...] Cargando modelo '{NOMBRE_MODELO}'...")

    tokenizador = AutoTokenizer.from_pretrained(NOMBRE_MODELO)

    # Cargar el modelo en CPU
    modelo = AutoModelForCausalLM.from_pretrained(NOMBRE_MODELO)
    modelo.eval()

    print("[OK] Modelo cargado correctamente.")
    return modelo, tokenizador


def responder(mensaje, historial):
    """Genera una respuesta a la pregunta del usuario. Sin memoria de conversacion.
    El parametro historial es requerido por ChatInterface pero no se usa.

    Args:
        mensaje: Texto ingresado por el usuario.
        historial: Ignorado. No se usa memoria.

    Returns:
        Respuesta generada por el modelo.
    """
    if not mensaje or not mensaje.strip():
        return "Por favor, escribe una pregunta."

    # Construir mensajes simples (sin historial, cada pregunta es independiente)
    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres un asistente virtual util y amigable. "
                "Responde de forma clara y concisa. "
                "Si te hablan en espanol, responde en espanol."
            ),
        },
        {"role": "user", "content": mensaje},
    ]

    # Preparar la entrada usando la plantilla de chat del modelo
    texto_entrada = tokenizador.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenizar la entrada
    entradas = tokenizador(
        [texto_entrada],
        return_tensors="pt",
    ).to(modelo.device)

    # Generar la respuesta
    salidas = modelo.generate(
        **entradas,
        max_new_tokens=MAX_TOKENS_RESPUESTA,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # Decodificar solo los tokens nuevos (sin el prompt)
    tokens_generados = salidas[0][entradas["input_ids"].shape[-1]:]
    respuesta = tokenizador.decode(tokens_generados, skip_special_tokens=True)

    return respuesta


# --- Punto de entrada principal ---
if __name__ == "__main__":
    # Cargar el modelo al iniciar
    modelo, tokenizador = cargar_modelo()

    # Interfaz estilo chat (similar a ChatGPT)
    interfaz = gr.ChatInterface(
        fn=responder,
        title="ChatBot Simple",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Escribe tu mensaje aqui...", scale=7),
    )

    # Lanzar la aplicacion
    print("\n[>>] Iniciando servidor...")
    interfaz.launch(
        server_name="127.0.0.1",  # Solo accesible desde este equipo
        server_port=7860,
        share=False,
        inbrowser=True,           # Abre el navegador automaticamente
    )
