import gradio as gr

from rag_pipeline import ask_question

def respond(message, history):

    if history is None:
        history = []

    # Add user message
    history.append({
        "role": "user",
        "content": message
    })

    # Generate bot response
    bot_message = ask_question(message)

    # Add assistant response
    history.append({
        "role": "assistant",
        "content": bot_message
    })

    return history, history

with gr.Blocks() as demo:

    gr.Markdown("# Airlines HR Assistant")

    chatbot = gr.Chatbot(
        type="messages",
        height=500
    )

    msg = gr.Textbox(
        placeholder="Ask HR policy questions..."
    )

    clear = gr.Button("Clear Chat")

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, chatbot]
    )

    clear.click(
        lambda: [],
        outputs=chatbot,
        queue=False
    )

demo.launch()
