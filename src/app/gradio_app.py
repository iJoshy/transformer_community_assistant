from __future__ import annotations

import gradio as gr

from evals.report import build_report

from .controller import (
    ChatSessionController,
)


APP_CSS = """
/* ============================================================
   Google Fonts – Inter for clean, modern typography
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ============================================================
   Design Tokens
   ============================================================ */
:root {
  --surface: #f5f1ea;
  --surface-strong: #ffffff;
  --accent: #0e7c86;
  --accent-dark: #0a5c63;
  --accent-soft: #e0f3f1;
  --accent-glow: rgba(14, 124, 134, 0.12);
  --ink: #1a1a2e;
  --ink-secondary: #2d3748;
  --muted: #4a5568;
  --edge: rgba(14, 124, 134, 0.18);
  --radius-lg: 22px;
  --radius-md: 16px;
  --radius-sm: 12px;
  --shadow-sm: 0 4px 14px rgba(0, 0, 0, 0.06);
  --shadow-md: 0 12px 32px rgba(14, 124, 134, 0.10);
  --shadow-lg: 0 20px 60px rgba(14, 124, 134, 0.15);
}

/* ============================================================
   Animations
   ============================================================ */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

/* ============================================================
   Global Styles
   ============================================================ */
body, .gradio-container {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  background:
    radial-gradient(ellipse at 10% 0%, rgba(14, 124, 134, 0.08) 0%, transparent 50%),
    radial-gradient(ellipse at 90% 0%, rgba(250, 204, 140, 0.12) 0%, transparent 40%),
    radial-gradient(ellipse at 50% 100%, rgba(14, 124, 134, 0.05) 0%, transparent 50%),
    linear-gradient(180deg, #f0ece4 0%, #f6f3ed 40%, #faf8f4 100%);
  color: var(--ink) !important;
}

/* ============================================================
   App Shell
   ============================================================ */
.app-shell {
  max-width: 1300px;
  margin: 0 auto;
  padding: 0 16px;
  animation: fadeInUp 0.5s ease-out;
}

/* ============================================================
   Hero Card
   ============================================================ */
.hero-card {
  background:
    linear-gradient(135deg, #0a5c63 0%, #0e7c86 40%, #11959f 70%, #0e7c86 100%);
  color: #ffffff;
  border-radius: var(--radius-lg);
  padding: 32px 34px 26px 34px;
  box-shadow: var(--shadow-lg);
  position: relative;
  overflow: hidden;
}

.hero-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.06) 40%,
    rgba(255, 255, 255, 0.12) 50%,
    rgba(255, 255, 255, 0.06) 60%,
    transparent 100%
  );
  background-size: 200% 100%;
  animation: shimmer 6s ease-in-out infinite;
  pointer-events: none;
}

.hero-card h1 {
  margin: 0 0 10px 0;
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #ffffff !important;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.hero-card p {
  margin: 0;
  max-width: 780px;
  line-height: 1.65;
  font-size: 1.05rem;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.95) !important;
}

/* ============================================================
   Panel Cards (status, feedback, etc.)
   ============================================================ */
.panel-card {
  background: var(--surface-strong) !important;
  border: 1.5px solid var(--edge) !important;
  border-radius: var(--radius-lg) !important;
  padding: 20px 22px !important;
  box-shadow: var(--shadow-md) !important;
  animation: fadeInUp 0.6s ease-out;
}

/* Force ALL text inside panels to be dark and readable */
.panel-card,
.panel-card *,
.panel-card h1, .panel-card h2, .panel-card h3,
.panel-card h4, .panel-card h5, .panel-card h6,
.panel-card p, .panel-card li, .panel-card span,
.panel-card strong, .panel-card em, .panel-card code {
  color: var(--ink) !important;
}

.panel-card h3 {
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  margin-bottom: 8px !important;
}

.panel-card li {
  font-size: 0.95rem !important;
  line-height: 1.6 !important;
}

.panel-card code {
  background: var(--accent-soft) !important;
  padding: 2px 7px !important;
  border-radius: 6px !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  color: var(--accent-dark) !important;
}

/* ============================================================
   Chatbot Column
   ============================================================ */
.chatbot-shell {
  overflow: visible;
}

.chatbot-shell .wrap {
  border-radius: var(--radius-lg);
}

/* Force chatbot container and messages area to have a light background */
.chatbot-shell .chatbot,
.chatbot-shell .chatbot > div,
.chatbot-shell .messages-wrapper,
.chatbot-shell .message-wrap,
.chatbot-shell [class*="chatbot"],
.chatbot-shell [data-testid="chatbot"],
.chatbot-shell [role="log"],
.chatbot-shell .wrap,
.chatbot-shell .wrap > div {
  background: #ffffff !important;
  background-color: #ffffff !important;
}

/* Ensure chatbot messages have readable dark text */
.chatbot-shell .message,
.chatbot-shell .message *,
.chatbot-shell .bot,
.chatbot-shell .bot *,
.chatbot-shell .user,
.chatbot-shell .user *,
.chatbot-shell p,
.chatbot-shell span {
  color: var(--ink) !important;
}

/* User message bubble - slightly tinted */
.chatbot-shell .user .message-bubble-border,
.chatbot-shell .user .message-content {
  background: var(--accent-soft) !important;
  color: var(--ink) !important;
}

/* Bot message bubble - white */
.chatbot-shell .bot .message-bubble-border,
.chatbot-shell .bot .message-content {
  background: #f8f9fa !important;
  color: var(--ink) !important;
}

/* Chatbot label */
.chatbot-shell label,
.chatbot-shell .label-wrap span {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
}

/* Chatbot empty state / placeholder */
.chatbot-shell .placeholder,
.chatbot-shell .empty {
  background: #ffffff !important;
  color: var(--muted) !important;
}

/* ============================================================
   Textbox Input
   ============================================================ */
.app-shell textarea,
.app-shell input[type="text"] {
  font-family: 'Inter', sans-serif !important;
  color: var(--ink) !important;
  background: var(--surface-strong) !important;
  border: 1.5px solid var(--edge) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.95rem !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.app-shell textarea:focus,
.app-shell input[type="text"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
  outline: none !important;
}

.app-shell textarea::placeholder {
  color: var(--muted) !important;
  opacity: 0.7;
}

/* Textbox labels */
.app-shell .input-label,
.app-shell label span {
  color: var(--ink) !important;
  font-weight: 600 !important;
}

/* ============================================================
   Buttons – Send & Clear
   ============================================================ */
.app-shell button.primary {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 28px !important;
  font-size: 0.95rem !important;
  box-shadow: 0 4px 16px rgba(14, 124, 134, 0.25) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.app-shell button.primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 22px rgba(14, 124, 134, 0.35) !important;
}

.app-shell button.primary:active {
  transform: translateY(0) !important;
}

.app-shell button.secondary,
.app-shell button:not(.primary):not(.example-btn) {
  color: var(--ink) !important;
  font-weight: 500 !important;
  border: 1.5px solid var(--edge) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--surface-strong) !important;
  transition: background 0.2s ease, border-color 0.2s ease !important;
}

.app-shell button.secondary:hover,
.app-shell button:not(.primary):not(.example-btn):hover {
  background: var(--accent-soft) !important;
  border-color: var(--accent) !important;
}

/* ============================================================
   Right-side Response Payload (JSON Panel)
   ============================================================ */
.response-payload {
  max-height: 440px;
  overflow-y: auto;
  border-radius: var(--radius-md) !important;
  animation: fadeInUp 0.6s ease-out;
  background: var(--surface-strong) !important;
  border: 1.5px solid var(--edge) !important;
  box-shadow: var(--shadow-md) !important;
}

/* Force ALL text inside JSON panel to be dark */
.response-payload,
.response-payload *,
.response-payload .json-holder,
.response-payload .json-holder *,
.response-payload pre,
.response-payload pre *,
.response-payload code,
.response-payload code *,
.response-payload span,
.response-payload div {
  color: var(--ink) !important;
}

/* Gradio JSON component internal structure overrides */
.response-payload .json-node,
.response-payload .json-node *,
.response-payload .json-value,
.response-payload .json-key,
.response-payload .json-string,
.response-payload .json-number,
.response-payload .json-boolean,
.response-payload .json-null,
.response-payload .json-toggle,
.response-payload .json-bracket {
  color: var(--ink) !important;
  opacity: 1 !important;
}

/* JSON line numbers */
.response-payload .line-number,
.response-payload .line-numbers,
.response-payload td:first-child {
  color: var(--muted) !important;
  opacity: 1 !important;
}

/* JSON container background */
.response-payload .json-holder,
.response-payload .json-container,
.response-payload table,
.response-payload tbody,
.response-payload tr,
.response-payload td {
  background: var(--surface-strong) !important;
}

/* JSON label */
.response-payload label,
.response-payload .label-wrap,
.response-payload .label-wrap span {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
}

/* Copy button in JSON panel */
.response-payload button {
  color: var(--muted) !important;
  opacity: 0.7;
}

.response-payload button:hover {
  opacity: 1;
}

/* ============================================================
   Support / Helper Text
   ============================================================ */
.support-copy {
  color: var(--muted) !important;
  font-size: 0.88rem !important;
  line-height: 1.55 !important;
  padding: 8px 0 !important;
}

.support-copy * {
  color: var(--muted) !important;
}

/* ============================================================
   Prompt Ideas & Feedback Panel – Force Readable Text
   ============================================================ */
.prompt-ideas,
.prompt-ideas * {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
}

.feedback-panel,
.feedback-panel *,
.feedback-panel p,
.feedback-panel span {
  color: var(--ink) !important;
}

/* ============================================================
   Example Button Row
   ============================================================ */
.example-row {
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 4px;
}

.example-row button {
  white-space: normal;
  word-break: break-word;
  text-align: left;
  background: var(--accent-soft) !important;
  color: var(--accent-dark) !important;
  border: 1.5px solid rgba(14, 124, 134, 0.22) !important;
  border-radius: 28px !important;
  padding: 8px 18px !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 8px rgba(14, 124, 134, 0.08) !important;
}

.example-row button:hover {
  background: rgba(14, 124, 134, 0.14) !important;
  border-color: var(--accent) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(14, 124, 134, 0.16) !important;
  color: var(--accent-dark) !important;
}

/* ============================================================
   Global Gradio Overrides – Ensure All Labels & Text Visible
   ============================================================ */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container .block label span {
  color: var(--ink) !important;
}

/* Markdown rendered inside any block */
.gradio-container .prose,
.gradio-container .prose * {
  color: var(--ink) !important;
}

/* Ensure tab labels and accordion headers are visible */
.gradio-container .tab-nav button,
.gradio-container .accordion .label-wrap {
  color: var(--ink) !important;
}

/* ============================================================
   Scrollbar Styling
   ============================================================ */
.response-payload::-webkit-scrollbar {
  width: 6px;
}

.response-payload::-webkit-scrollbar-track {
  background: transparent;
}

.response-payload::-webkit-scrollbar-thumb {
  background: rgba(14, 124, 134, 0.2);
  border-radius: 3px;
}

.response-payload::-webkit-scrollbar-thumb:hover {
  background: rgba(14, 124, 134, 0.35);
}
"""


DEFAULT_EXAMPLES = [
    "What events are coming up this month?",
    "Register me for Community Meetup. My email is user@example.com.",
    "Which events am I registered for? My email is user@example.com.",
]


def clear_input() -> str:
    return ""


def _format_online_summary() -> str:
    """Load online eval logs and return a styled markdown summary."""
    try:
        report = build_report()
        summary = report.get("online_summary", {})
    except Exception:
        return (
            "### \u26a0\ufe0f No eval data available\n\n"
            "Start chatting and giving feedback (thumbs up/down) to generate online eval signals."
        )

    total_responses = summary.get("total_responses", 0)
    if total_responses == 0:
        return (
            "### \u26a0\ufe0f No eval data available\n\n"
            "Start chatting and giving feedback (thumbs up/down) to generate online eval signals."
        )

    total_feedback = summary.get("total_feedback", 0)
    likes = summary.get("likes", 0)
    dislikes = summary.get("dislikes", 0)
    approval = summary.get("approval_rate", 0.0)
    disapproval = summary.get("disapproval_rate", 0.0)
    coverage = summary.get("feedback_coverage_rate", 0.0)
    retrieval_rate = summary.get("retrieval_usage_rate", 0.0)
    tool_call_rate = summary.get("tool_call_rate", 0.0)
    tool_success_rate = summary.get("tool_success_rate", 0.0)
    avg_latency = summary.get("average_latency_ms", 0.0)

    approval_emoji = "\u2705" if approval >= 0.8 else ("\u26a0\ufe0f" if approval >= 0.5 else "\u274c")
    latency_emoji = "\u26a1" if avg_latency < 3000 else ("\u23f3" if avg_latency < 5000 else "\U0001f422")

    return f"""### \U0001f4ca Online Eval Summary

---

#### User Feedback
| Metric | Value |
|--------|-------|
| Total responses | **{total_responses}** |
| Total feedback | **{total_feedback}** |
| Feedback coverage | **{coverage:.1%}** |
| {approval_emoji} Approval rate | **{approval:.1%}** |
| Likes | **{likes}** \U0001f44d |
| Dislikes | **{dislikes}** \U0001f44e |

---

#### Retrieval & Tool Usage
| Metric | Value |
|--------|-------|
| Retrieval usage rate | **{retrieval_rate:.1%}** |
| Tool call rate | **{tool_call_rate:.1%}** |
| Tool success rate | **{tool_success_rate:.1%}** |

---

#### Performance
| Metric | Value |
|--------|-------|
| {latency_emoji} Avg latency | **{avg_latency:,.0f} ms** |
"""


def build_demo(
    controller: ChatSessionController,
    *,
    title: str = "Transformer Community Assistant",
    examples: list[str] | None = None,
) -> gr.Blocks:
    examples = examples or list(DEFAULT_EXAMPLES)
    _, initial_status, initial_details, initial_session_state, initial_feedback_status = controller.reset()

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="amber",
            neutral_hue="stone",
        ),
        css=APP_CSS,
    ) as demo:
        session_state = gr.State(value=initial_session_state)
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                <div class="hero-card">
                  <h1>Transformer Community Assistant</h1>
                  <p>
                    Your AI-powered community hub ask questions, register for events, or look up your registrations.
                  </p>
                </div>
                """
            )

            with gr.Tabs():
                with gr.Tab("\U0001f4ac Chat"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["panel-card", "chatbot-shell"]):
                            chatbot = gr.Chatbot(
                                value=[],
                                type="messages",
                                label="Community chat",
                                height=580,
                                feedback_options=("Like", "Dislike"),
                                allow_tags=False,
                            )
                            message = gr.Textbox(
                                label="Message",
                                placeholder="Ask about events, register for one, or check registrations by email...",
                                lines=3,
                                max_lines=6,
                            )
                            with gr.Row():
                                send_button = gr.Button("Send", variant="primary")
                                clear_button = gr.Button("Clear chat")
                            gr.Markdown("**Prompt ideas**", elem_classes=["prompt-ideas"])
                            with gr.Row(elem_classes=["example-row"]):
                                example_buttons = [
                                    gr.Button(example, size="sm")
                                    for example in examples
                                ]

                        with gr.Column(scale=4):
                            status = gr.Markdown(
                                value=initial_status,
                                elem_classes=["panel-card"],
                            )
                            details = gr.JSON(
                                value=initial_details,
                                label="Latest response payload",
                                elem_classes=["response-payload"],
                            )
                            feedback_status = gr.Markdown(
                                value=initial_feedback_status,
                                elem_classes=["panel-card", "feedback-panel"],
                            )
                            gr.Markdown(
                                """
                                <div class="support-copy">
                                  The panel on the right shows retrieval status, tool usage, and the raw assistant payload.
                                  This keeps the UI user-friendly while preserving the backend details needed for debugging and later evals.
                                </div>
                                """
                            )

                    submit_event = message.submit(
                        controller.handle_message,
                        inputs=[message, chatbot, session_state],
                        outputs=[chatbot, status, details, session_state, feedback_status],
                    )
                    send_event = send_button.click(
                        controller.handle_message,
                        inputs=[message, chatbot, session_state],
                        outputs=[chatbot, status, details, session_state, feedback_status],
                    )
                    submit_event.then(clear_input, outputs=message)
                    send_event.then(clear_input, outputs=message)
                    clear_button.click(
                        controller.reset,
                        outputs=[chatbot, status, details, session_state, feedback_status],
                        queue=False,
                    )

                    def handle_feedback(
                        history,
                        stored_session_state,
                        like_data: gr.LikeData,
                    ):
                        return controller.handle_feedback(history, stored_session_state, like_data)

                    chatbot.like(
                        handle_feedback,
                        inputs=[chatbot, session_state],
                        outputs=[session_state, feedback_status],
                        queue=False,
                        show_api=False,
                    )
                    for index, example_button in enumerate(example_buttons):
                        example_button.click(
                            lambda selected=index: examples[selected],
                            outputs=message,
                            queue=False,
                        )

                with gr.Tab("\U0001f4ca Eval Dashboard"):
                    gr.Markdown("### Live Online Eval Metrics", elem_classes=["prompt-ideas"])
                    gr.Markdown(
                        "Click **Refresh** to load the latest feedback and performance metrics from your session logs.",
                        elem_classes=["support-copy"],
                    )
                    eval_display = gr.Markdown(
                        value=_format_online_summary(),
                        elem_classes=["panel-card"],
                    )
                    refresh_button = gr.Button("\U0001f504 Refresh Eval Summary", variant="primary")
                    refresh_button.click(
                        _format_online_summary,
                        outputs=[eval_display],
                        queue=False,
                    )

    return demo

