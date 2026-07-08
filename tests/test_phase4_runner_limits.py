import unittest
from pathlib import Path

from cial_knowledge_os.config import Phase4Config
from cial_knowledge_os.phase4_runner import Phase4Runner


class Phase4RunnerLimitTests(unittest.TestCase):
    def test_default_manual_questions_are_not_trimmed(self) -> None:
        config = Phase4Config(project_root=Path("."))
        runner = object.__new__(Phase4Runner)
        runner.config = config

        questions = [f"question {index}" for index in range(10)]
        self.assertEqual(
            list(runner._apply_mode_limits(questions, run_mode="manual_qa")),
            questions,
        )

    def test_unlimited_manual_questions_are_not_trimmed(self) -> None:
        config = Phase4Config(
            project_root=Path("."),
            max_inline_manual_questions=None,
            allow_large_run=False,
        )
        runner = object.__new__(Phase4Runner)
        runner.config = config

        questions = [f"question {index}" for index in range(10)]
        self.assertEqual(
            list(runner._apply_mode_limits(questions, run_mode="manual_qa")),
            questions,
        )


if __name__ == "__main__":
    unittest.main()
