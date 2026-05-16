import gradio as gr

from rag_pipeline import ask_question

def respond(message, history):

    if history is None:
        history = []

    response = ask_question(message)

    history.append(
        (message, response)
    )

    return "", history

with gr.Blocks() as demo:

    gr.Markdown("# Airlines HR Assistant")

    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(
        placeholder="Ask a question..."
    )

    clear = gr.Button("Clear Chat")

    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot]
    )

    clear.click(
        lambda: None,
        None,
        chatbot,
        queue=False
    )

demo.launch()
