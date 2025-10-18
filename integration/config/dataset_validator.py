"""
Dataset format validator - PRODUCTION READY.
Enforces OpenAI-style conversation format with strict validation rules.

Validation Rules:
1. Dataset must use OpenAI format (role + content)
2. Each conversation must have at least 1 assistant message
3. Conversations MUST end with assistant role (not user)
4. Content cannot be null or empty
5. Content can be string OR list (for multimodal: text, image_url types)
6. Role must be one of: system, user, assistant
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class OpenAIDatasetValidator:
    """
    Validates datasets are in OpenAI conversation format.

    OpenAI Format:
    {
        "conversations": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"}
        ]
    }

    Or with multimodal content:
    {
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://..."}}
                ]
            },
            {"role": "assistant", "content": "I see a cat"}
        ]
    }
    """

    VALID_ROLES = {"system", "user", "assistant"}
    VALID_CONTENT_TYPES = {"text", "image_url"}

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, fails on any error. If False, only warns.
        """
        self.strict_mode = strict_mode
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats = {
            "total_conversations": 0,
            "total_messages": 0,
            "role_counts": {"system": 0, "user": 0, "assistant": 0},
            "avg_messages_per_conversation": 0.0,
            "conversations_ending_with_user": 0,
            "empty_content_found": 0,
            "invalid_roles": 0
        }

    def validate_dataset(self, dataset_path: str) -> ValidationResult:
        """
        Validate entire dataset file.

        Args:
            dataset_path: Path to dataset file (.jsonl or .json)

        Returns:
            ValidationResult with validation details
        """
        logger.info(f"Validating dataset: {dataset_path}")

        path = Path(dataset_path)
        if not path.exists():
            self.errors.append(f"Dataset file not found: {dataset_path}")
            return self._build_result()

        # Check file extension
        if path.suffix not in ['.jsonl', '.json']:
            self.errors.append(
                f"Dataset must be .jsonl or .json file, got: {path.suffix}"
            )
            return self._build_result()

        # Load and validate
        try:
            if path.suffix == '.jsonl':
                self._validate_jsonl(path)
            else:
                self._validate_json(path)
        except Exception as e:
            self.errors.append(f"Failed to validate dataset: {e}")
            logger.exception("Dataset validation failed")

        return self._build_result()

    def _validate_jsonl(self, path: Path) -> None:
        """Validate JSONL dataset (one conversation per line)."""
        line_num = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()

                if not line:  # Skip empty lines
                    continue

                try:
                    data = json.loads(line)
                    self._validate_conversation(data, f"line {line_num}")
                except json.JSONDecodeError as e:
                    self.errors.append(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    self.errors.append(f"Error at line {line_num}: {e}")

        if line_num == 0:
            self.errors.append("Dataset file is empty")

    def _validate_json(self, path: Path) -> None:
        """Validate JSON dataset (array of conversations)."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            self.errors.append(
                "JSON dataset must be a list of conversations"
            )
            return

        for idx, conversation in enumerate(data):
            self._validate_conversation(conversation, f"conversation {idx}")

    def _validate_conversation(self, data: Dict[str, Any], context: str) -> None:
        """
        Validate single conversation.

        Args:
            data: Conversation data
            context: Context string for error messages (e.g., "line 5")
        """
        self.stats["total_conversations"] += 1

        # Check for 'conversations' key
        if "conversations" not in data:
            self.errors.append(
                f"{context}: Missing 'conversations' key. "
                f"Expected format: {{'conversations': [{{'role': 'user', 'content': '...'}}]}}"
            )
            return

        conversations = data["conversations"]

        # Validate conversations is a list
        if not isinstance(conversations, list):
            self.errors.append(
                f"{context}: 'conversations' must be a list, got {type(conversations)}"
            )
            return

        # Validate not empty
        if len(conversations) == 0:
            self.errors.append(f"{context}: 'conversations' list is empty")
            return

        # Validate each message
        assistant_count = 0
        last_role = None

        for msg_idx, message in enumerate(conversations):
            msg_context = f"{context}, message {msg_idx}"

            # Validate message structure
            validation_errors = self._validate_message(message, msg_context)
            if validation_errors:
                self.errors.extend(validation_errors)
                continue

            # Track stats
            role = message["role"]
            self.stats["role_counts"][role] = self.stats["role_counts"].get(role, 0) + 1
            self.stats["total_messages"] += 1

            if role == "assistant":
                assistant_count += 1

            last_role = role

        # RULE: At least 1 assistant message required
        if assistant_count == 0:
            self.errors.append(
                f"{context}: No assistant messages found. "
                f"Each conversation must have at least 1 assistant message."
            )

        # RULE: Conversation MUST end with assistant
        if last_role != "assistant":
            self.errors.append(
                f"{context}: Conversation ends with '{last_role}' role. "
                f"REQUIRED: Conversations must end with 'assistant' role."
            )
            self.stats["conversations_ending_with_user"] += 1

    def _validate_message(
        self,
        message: Dict[str, Any],
        context: str
    ) -> List[str]:
        """
        Validate single message.

        Args:
            message: Message dictionary
            context: Context for error messages

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required keys
        if "role" not in message:
            errors.append(f"{context}: Missing 'role' key")
            return errors

        if "content" not in message:
            errors.append(f"{context}: Missing 'content' key")
            return errors

        # Validate role
        role = message["role"]
        if role not in self.VALID_ROLES:
            errors.append(
                f"{context}: Invalid role '{role}'. "
                f"Must be one of: {', '.join(self.VALID_ROLES)}"
            )
            self.stats["invalid_roles"] += 1

        # Validate content
        content = message["content"]
        content_errors = self._validate_content(content, context)
        errors.extend(content_errors)

        return errors

    def _validate_content(
        self,
        content: Union[str, List, None],
        context: str
    ) -> List[str]:
        """
        Validate message content.

        Content can be:
        1. String: "Hello world"
        2. List of content items: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]

        Args:
            content: Content value
            context: Context for error messages

        Returns:
            List of validation errors
        """
        errors = []

        # RULE: Content cannot be null
        if content is None:
            errors.append(f"{context}: Content is null. Content cannot be null.")
            self.stats["empty_content_found"] += 1
            return errors

        # RULE: Content cannot be empty
        if isinstance(content, str):
            if not content.strip():
                errors.append(f"{context}: Content is empty string")
                self.stats["empty_content_found"] += 1

        elif isinstance(content, list):
            # Validate list content (multimodal)
            if len(content) == 0:
                errors.append(f"{context}: Content list is empty")
                self.stats["empty_content_found"] += 1
                return errors

            # Validate each content item
            for item_idx, item in enumerate(content):
                if not isinstance(item, dict):
                    errors.append(
                        f"{context}, content[{item_idx}]: Content item must be a dict"
                    )
                    continue

                # Check type field
                if "type" not in item:
                    errors.append(
                        f"{context}, content[{item_idx}]: Missing 'type' field"
                    )
                    continue

                item_type = item["type"]
                if item_type not in self.VALID_CONTENT_TYPES:
                    errors.append(
                        f"{context}, content[{item_idx}]: Invalid type '{item_type}'. "
                        f"Must be one of: {', '.join(self.VALID_CONTENT_TYPES)}"
                    )

                # Validate content based on type
                if item_type == "text":
                    if "text" not in item:
                        errors.append(
                            f"{context}, content[{item_idx}]: Missing 'text' field for type='text'"
                        )
                    elif not item["text"] or not item["text"].strip():
                        errors.append(
                            f"{context}, content[{item_idx}]: Text content is empty"
                        )

                elif item_type == "image_url":
                    if "image_url" not in item:
                        errors.append(
                            f"{context}, content[{item_idx}]: Missing 'image_url' field for type='image_url'"
                        )
                    elif not isinstance(item["image_url"], dict):
                        errors.append(
                            f"{context}, content[{item_idx}]: 'image_url' must be a dict"
                        )
                    elif "url" not in item.get("image_url", {}):
                        errors.append(
                            f"{context}, content[{item_idx}]: Missing 'url' in image_url"
                        )

        else:
            errors.append(
                f"{context}: Content must be string or list, got {type(content)}"
            )

        return errors

    def _build_result(self) -> ValidationResult:
        """Build final validation result."""
        # Calculate stats
        if self.stats["total_conversations"] > 0:
            self.stats["avg_messages_per_conversation"] = (
                self.stats["total_messages"] / self.stats["total_conversations"]
            )

        is_valid = len(self.errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            stats=self.stats
        )


def validate_dataset(
    dataset_path: str,
    strict_mode: bool = True
) -> ValidationResult:
    """
    Validate dataset is in correct OpenAI format.

    Args:
        dataset_path: Path to dataset file
        strict_mode: If True, fails on any error

    Returns:
        ValidationResult with validation details

    Raises:
        ValueError: If validation fails and strict_mode is True

    Example:
        >>> result = validate_dataset("data.jsonl")
        >>> if not result.is_valid:
        >>>     for error in result.errors:
        >>>         print(f"ERROR: {error}")
    """
    validator = OpenAIDatasetValidator(strict_mode=strict_mode)
    result = validator.validate_dataset(dataset_path)

    # Log results
    if result.is_valid:
        logger.info("✅ Dataset validation PASSED")
        logger.info(f"Total conversations: {result.stats['total_conversations']}")
        logger.info(f"Total messages: {result.stats['total_messages']}")
        logger.info(f"Role distribution: {result.stats['role_counts']}")
    else:
        logger.error(f"❌ Dataset validation FAILED with {len(result.errors)} errors")
        for error in result.errors:
            logger.error(f"  - {error}")

    # Log warnings
    for warning in result.warnings:
        logger.warning(f"  - {warning}")

    # Raise if strict mode and errors found
    if strict_mode and not result.is_valid:
        error_summary = "\n".join(f"  - {e}" for e in result.errors[:10])
        if len(result.errors) > 10:
            error_summary += f"\n  ... and {len(result.errors) - 10} more errors"

        raise ValueError(
            f"Dataset validation failed with {len(result.errors)} errors:\n{error_summary}\n\n"
            f"Dataset format requirements:\n"
            f"1. Use OpenAI format with 'conversations' key\n"
            f"2. Each message needs 'role' (system/user/assistant) and 'content'\n"
            f"3. At least 1 assistant message per conversation\n"
            f"4. Conversations MUST end with assistant role\n"
            f"5. Content cannot be null or empty\n"
            f"6. Content can be string or list (for multimodal)\n"
        )

    return result


def quick_validate(dataset_path: str, max_samples: int = 100) -> bool:
    """
    Quick validation of first N samples.

    Args:
        dataset_path: Path to dataset
        max_samples: Number of samples to validate

    Returns:
        True if valid, False otherwise
    """
    logger.info(f"Quick validation of {max_samples} samples from {dataset_path}")

    path = Path(dataset_path)
    if not path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False

    try:
        # Read first N lines
        sample_count = 0
        errors = []

        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if sample_count >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Basic checks
                    if "conversations" not in data:
                        errors.append(f"Line {line_num}: Missing 'conversations'")
                        continue

                    conversations = data["conversations"]
                    if not conversations:
                        errors.append(f"Line {line_num}: Empty conversations")
                        continue

                    # Check last message is assistant
                    if conversations[-1].get("role") != "assistant":
                        errors.append(
                            f"Line {line_num}: Must end with assistant, got '{conversations[-1].get('role')}'"
                        )

                    sample_count += 1

                except json.JSONDecodeError:
                    errors.append(f"Line {line_num}: Invalid JSON")

        if errors:
            logger.error(f"Quick validation found {len(errors)} errors:")
            for error in errors[:5]:
                logger.error(f"  - {error}")
            return False

        logger.info(f"✅ Quick validation passed ({sample_count} samples)")
        return True

    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False


def generate_example_dataset(output_path: str, num_samples: int = 10) -> None:
    """
    Generate example dataset in correct OpenAI format.

    Args:
        output_path: Where to save example dataset
        num_samples: Number of sample conversations to generate
    """
    examples = []

    for i in range(num_samples):
        # Mix of different conversation patterns
        if i % 3 == 0:
            # Simple user-assistant exchange
            examples.append({
                "conversations": [
                    {"role": "user", "content": f"Question {i}?"},
                    {"role": "assistant", "content": f"Answer {i}."}
                ]
            })
        elif i % 3 == 1:
            # With system message
            examples.append({
                "conversations": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Help me with task {i}"},
                    {"role": "assistant", "content": f"Sure! Here's how to do task {i}..."}
                ]
            })
        else:
            # Multi-turn conversation
            examples.append({
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help?"},
                    {"role": "user", "content": f"Tell me about topic {i}"},
                    {"role": "assistant", "content": f"Topic {i} is interesting because..."}
                ]
            })

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    logger.info(f"✅ Generated {num_samples} example conversations at {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_validator.py <dataset.jsonl>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    result = validate_dataset(dataset_path, strict_mode=False)

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"\nStatistics:")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")

    if not result.is_valid:
        sys.exit(1)
