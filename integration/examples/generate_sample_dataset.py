"""
Generate sample dataset in OpenAI format for testing.
This creates a valid dataset that passes all validation rules.
"""

import json
import argparse
from pathlib import Path


def generate_sample_conversations(num_samples=100):
    """Generate sample conversations in OpenAI format."""

    conversations = []

    # Templates for variety
    templates = [
        # Simple QA
        {
            "conversations": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}
            ]
        },
        # With system message
        {
            "conversations": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain neural networks"},
                {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information."}
            ]
        },
        # Multi-turn
        {
            "conversations": [
                {"role": "user", "content": "I'm learning Python"},
                {"role": "assistant", "content": "That's great! Python is an excellent language. What would you like to learn?"},
                {"role": "user", "content": "How do I create a list?"},
                {"role": "assistant", "content": "You can create a list using square brackets: my_list = [1, 2, 3]"}
            ]
        },
        # Code generation
        {
            "conversations": [
                {"role": "system", "content": "You are an expert programmer."},
                {"role": "user", "content": "Write a function to reverse a string"},
                {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"}
            ]
        },
        # Longer conversation
        {
            "conversations": [
                {"role": "user", "content": "I need help with my project"},
                {"role": "assistant", "content": "I'd be happy to help! What kind of project are you working on?"},
                {"role": "user", "content": "A web application"},
                {"role": "assistant", "content": "Great! What framework are you using?"},
                {"role": "user", "content": "React"},
                {"role": "assistant", "content": "React is excellent for web apps. What specific help do you need?"}
            ]
        }
    ]

    # Generate num_samples by cycling through templates
    for i in range(num_samples):
        template = templates[i % len(templates)]

        # Customize with sample number
        customized = {
            "conversations": [
                {
                    "role": msg["role"],
                    "content": msg["content"] + f" [Sample {i}]" if i < 5 else msg["content"]
                }
                for msg in template["conversations"]
            ]
        }

        conversations.append(customized)

    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample OpenAI-format dataset for testing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_dataset.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of conversations to generate"
    )

    args = parser.parse_args()

    # Generate conversations
    print(f"Generating {args.num_samples} sample conversations...")
    conversations = generate_sample_conversations(args.num_samples)

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

    print(f"✅ Generated {len(conversations)} conversations")
    print(f"📁 Saved to: {output_path}")
    print(f"\nValidation rules enforced:")
    print("  ✓ OpenAI format (role + content)")
    print("  ✓ All conversations end with assistant")
    print("  ✓ At least 1 assistant message per conversation")
    print("  ✓ No null or empty content")
    print(f"\nYou can now use this dataset in your config:")
    print(f"  dataset_uri: \"{output_path.absolute()}\"")

    # Show first conversation as example
    print(f"\nExample conversation:")
    print(json.dumps(conversations[0], indent=2))


if __name__ == "__main__":
    main()
