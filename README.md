# 🤖 ChatBot Simple

Chatbot local impulsado por **Qwen2.5-0.5B-Instruct** con interfaz web mediante **Gradio**.

## Requisitos

- Python 3.9 o superior
- ~2 GB de RAM disponible
- ~1 GB de espacio en disco (para el modelo)

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate    # Windows

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

```bash
python app.py
```

La primera ejecución descargará el modelo (~1 GB). Las siguientes serán más rápidas.

Una vez cargado, abre tu navegador en: **http://localhost:7860**

## Modelo

- **Nombre:** Qwen/Qwen2.5-0.5B-Instruct
- **Tamaño:** ~0.5B parámetros
- **Tipo:** Modelo de instrucciones (chat)
- **Idiomas:** Multilingüe (incluye español e inglés)
- **Licencia:** Apache 2.0

## Características

- ✅ Funciona completamente en local (sin API externa)
- ✅ Interfaz web amigable con historial de chat
- ✅ Soporte multilingüe (español, inglés, etc.)
- ✅ Ejemplos de preguntas incluidos
- ✅ Corre en CPU (GPU opcional para mayor velocidad)
