"""
Prompt templates for grounded email reply generation.

The generator uses few-shot examples retrieved from the dataset pool
to ground its replies in real patterns, not isolated prompting.
"""

from data.schemas import EmailScenario


def _tone_instruction(sentiment: str) -> str:
    """Map customer sentiment to explicit tone guidance."""
    tone_map = {
        "angry": (
            "Use an empathetic and apologetic tone. Acknowledge the customer's "
            "frustration directly. Show urgency in resolving their issue."
        ),
        "confused": (
            "Use a clear, patient, and reassuring tone. Break down information "
            "into simple steps. Avoid jargon."
        ),
        "neutral": (
            "Use a professional and concise tone. Be efficient and direct — "
            "the customer wants a quick resolution, not emotional reassurance."
        ),
        "happy": (
            "Use a warm, professional tone. Match their positive energy while "
            "being helpful. Thank them for their engagement."
        ),
    }
    return tone_map.get(sentiment, tone_map["neutral"])


def _format_few_shot_example(example: EmailScenario) -> str:
    """Format a single few-shot example for the prompt."""
    thread_ctx = ""
    if example.prior_thread_summary:
        thread_ctx = f"\nPrior conversation: {example.prior_thread_summary}"

    return f"""--- Example ---
Customer email subject: {example.subject}
Customer email: {example.customer_email_body}
Category: {example.category} | Sentiment: {example.sentiment} | Urgency: {example.urgency}{thread_ctx}

Agent reply:
{example.reference_reply}
--- End Example ---"""


def build_reply_prompt(
    target: EmailScenario,
    few_shot_examples: list[EmailScenario],
) -> str:
    """
    Build the full generation prompt with system context, few-shot examples,
    and the target email.
    """
    # Format few-shot examples
    examples_block = "\n\n".join(
        _format_few_shot_example(ex) for ex in few_shot_examples
    )

    # Thread context for target
    thread_ctx = ""
    if target.prior_thread_summary:
        thread_ctx = (
            f"\nPrior conversation context: {target.prior_thread_summary}"
        )

    tone = _tone_instruction(target.sentiment)

    return f"""You are a customer support agent for a SaaS company that provides shared inbox and email collaboration tools. Your job is to write helpful, professional email replies.

Here are examples of past customer emails and the replies our team sent:

{examples_block}

Now write a reply to this new customer email:

Subject: {target.subject}
Customer email: {target.customer_email_body}
Category: {target.category}
Customer sentiment: {target.sentiment}
Urgency: {target.urgency}{thread_ctx}

Tone instruction: {tone}

Requirements:
- Address the customer's specific question or concern directly
- Include a concrete next step or action item
- Keep the reply between 80-180 words
- Do NOT invent specific policy details, refund amounts, pricing, or promises that are not grounded in the input context or the examples provided
- Do NOT fabricate PII, ticket numbers, or specific dates
- Be professional and human — not robotic

Respond with a JSON object matching the required schema."""
