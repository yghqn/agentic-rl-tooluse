"""Deterministic V1 tasks; no LLM, Agent, or trajectory generation."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import random
from typing import Any

from environment.database import SUPPORTED_FIELDS, SyntheticCompanyDatabase
from environment.tools import calculator
from tasks.schemas import Task
from tasks.validators import (
    KNOWN_TOOLS,
    environment_id_for_seed,
    verification_spec_for_type,
)


def agent_task_view(task: Task) -> dict[str, Any]:
    """Return only Agent-visible inputs, never the privileged Task record.

    Future prompt builders should use this allowlisted view rather than
    serializing Task, whose ground_truth and verification_spec are private
    evaluation data. Metadata is also deliberately excluded.
    """

    return {"question": task.question, "available_tools": list(task.available_tools)}


class TaskGenerator:
    """Generate tasks independently by (seed, difficulty, task_index).

    A repeated coordinate intentionally yields the same ID and task. Different
    indices yield distinct IDs, even if their natural-language questions match.
    """

    def __init__(self, seed: int = 0) -> None:
        self._database = SyntheticCompanyDatabase(seed=seed)
        self._seed = seed

    def generate_task(self, difficulty: int, task_index: int = 0) -> Task:
        """Generate one reproducible task with all three tools available."""

        if type(difficulty) is not int or difficulty not in {1, 2, 3, 4}:
            raise ValueError("difficulty must be one of 1, 2, 3, 4")
        if type(task_index) is not int or task_index < 0:
            raise ValueError("task_index must be a non-negative integer")

        coordinate = f"task-v1-seed-{self._seed}-l{difficulty}-index-{task_index}"
        derived_seed = int.from_bytes(
            hashlib.sha256(coordinate.encode("utf-8")).digest(), byteorder="big"
        )
        random_generator = random.Random(derived_seed)
        companies = self._database.list_companies()
        metadata: dict[str, Any]
        ground_truth: Any

        if difficulty == 1:
            task_type = "single_retrieval"
            company = random_generator.choice(companies)
            field = random_generator.choice(SUPPORTED_FIELDS)
            metadata = {"company": company, "field": field}
            question = f"What is the {field} of {company}?"
            ground_truth = self._database.lookup(company, field)
        elif difficulty == 2 and task_index % 2 == 0:
            task_type = "list_companies"
            metadata = {}
            question = "List all companies available in the current environment."
            ground_truth = companies
        elif difficulty == 2:
            task_type = "arithmetic"
            left = random_generator.randint(1, 99)
            right = random_generator.randint(1, 99)
            operation = random_generator.choice(("+", "-", "*", "/"))
            expression = f"{left} {operation} {right}"
            metadata = {"expression": expression}
            question = f"Calculate {expression}."
            ground_truth = calculator(expression)
        elif difficulty == 3:
            task_type = "profit_margin"
            company = random_generator.choice(companies)
            metadata = {"company": company, "fields": ["profit", "revenue"]}
            question = (
                f"What is the profit margin of {company}? "
                "Return profit / revenue as a ratio, not a percentage."
            )
            ground_truth = float(self._margin(company))
        else:
            task_type = "highest_profit_margin"
            selected = sorted(random_generator.sample(companies, 3))
            metadata = {"companies": selected, "fields": ["profit", "revenue"]}
            question = (
                f"Among {selected[0]}, {selected[1]}, and {selected[2]}, "
                "which company has the highest profit margin (profit / revenue)? "
                "Return the company name and its profit margin as a ratio. "
                "If there is a tie, choose the alphabetically first company."
            )
            margins = {company: self._margin(company) for company in selected}
            winner = min(margins, key=lambda company: (-margins[company], company))
            ground_truth = {"company": winner, "profit_margin": float(margins[winner])}

        return Task(
            task_id=coordinate,
            question=question,
            difficulty=difficulty,
            environment_id=environment_id_for_seed(self._seed),
            available_tools=list(KNOWN_TOOLS),
            ground_truth=ground_truth,
            verification_spec=verification_spec_for_type(task_type),
            metadata=metadata,
        )

    def generate_tasks(self, count_per_level: int) -> list[Task]:
        """Generate levels 1 through 4 in fixed level/index order.

        Use at least two tasks per level to include both L2 categories.
        """

        if type(count_per_level) is not int or count_per_level < 0:
            raise ValueError("count_per_level must be a non-negative integer")
        return [
            self.generate_task(difficulty, task_index)
            for difficulty in (1, 2, 3, 4)
            for task_index in range(count_per_level)
        ]

    def _margin(self, company: str) -> Fraction:
        return Fraction(
            self._database.lookup(company, "profit"),
            self._database.lookup(company, "revenue"),
        )
