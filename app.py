import gradio as gr

from rag_pipeline import ask_question

def respond(message, history):

    if history is None:
        history = []

    response = ask_question(message)

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    return history

with gr.Blocks() as demo:

    gr.Markdown("# Airlines HR Assistant")

    chatbot = gr.Chatbot(
        label="HR Chat",
        height=500
    )

    msg = gr.Textbox(
        placeholder="Ask HR policy questions..."
    )

    clear = gr.Button("Clear Chat")

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    )

    clear.click(
        lambda: [],
        outputs=[chatbot],
        queue=False
    )

demo.launch()
