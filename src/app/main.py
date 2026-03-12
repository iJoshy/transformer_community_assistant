from __future__ import annotations

import argparse
from typing import Sequence

from assistant import CommunityAssistantService
from evals import OnlineFeedbackLogger
from model_provider import get_default_chat_model, get_default_embedding_model
from rag import RagConfig
from runtime_env import ensure_dotenv_loaded

from .controller import ChatSessionController
from .gradio_app import build_demo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Transformer Community Assistant Gradio app."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share links.")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the app in a browser.")
    parser.add_argument("--persist-dir", default="vector_db", help="Chroma persistence directory.")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model used to load the vector store. Defaults by provider env.",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Chat model used by the assistant. Defaults by provider env.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Chat model temperature.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve per question.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Maximum retrieved context length.",
    )
    parser.add_argument(
        "--response-log",
        default="data/evals/response_events.jsonl",
        help="Path to the online response event log.",
    )
    parser.add_argument(
        "--feedback-log",
        default="data/evals/feedback_events.jsonl",
        help="Path to the online feedback event log.",
    )
    return parser.parse_args(argv)


def create_controller(args: argparse.Namespace) -> ChatSessionController:
    ensure_dotenv_loaded()
    rag_config = RagConfig(
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model or get_default_embedding_model(),
    )
    assistant_service = CommunityAssistantService.from_env(
        rag_config=rag_config,
        model=args.chat_model or get_default_chat_model(),
        temperature=args.temperature,
    )
    feedback_logger = OnlineFeedbackLogger(
        response_log_path=args.response_log,
        feedback_log_path=args.feedback_log,
    )
    return ChatSessionController(
        assistant_service=assistant_service,
        k=args.k,
        max_chars=args.max_chars,
        online_feedback_logger=feedback_logger,
    )


def create_demo(args: argparse.Namespace):
    controller = create_controller(args)
    return build_demo(controller)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    demo = create_demo(args)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        inbrowser=args.inbrowser,
        show_api=False,
    )


if __name__ == "__main__":
    main()
