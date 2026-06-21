import argparse

from app.config import get_settings
from app.services.claude_corrector import ClaudeCorrector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="I WANT MEDSIN FROM STORE",
        help="Raw STT transcript to correct",
    )
    args = parser.parse_args()

    settings = get_settings()

    print("========== CLAUDE CORRECTOR TEST ==========")
    print(f"Claude model: {settings.ANTHROPIC_MODEL}")
    print(f"Raw transcript: {args.text}")

    corrector = ClaudeCorrector(
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.ANTHROPIC_MODEL,
    )

    corrected = corrector.correct(args.text)

    print("\n========== RESULT ==========")
    print(f"Corrected text: {corrected}")

    assert isinstance(corrected, str), "Corrected output should be a string"
    assert corrected.strip(), "Corrected output should not be empty"

    print("\nClaude corrector test passed.")


if __name__ == "__main__":
    main()