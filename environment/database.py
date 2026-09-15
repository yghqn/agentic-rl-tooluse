"""Deterministic synthetic company data used by the V1 environment."""

from __future__ import annotations

from dataclasses import dataclass
import random


COMPANY_NAMES: tuple[str, ...] = tuple(
    f"Company {letter}" for letter in "ABCDEFGHIJ"
)
SUPPORTED_FIELDS: tuple[str, ...] = (
    "revenue",
    "profit",
    "employees",
    "growth_rate",
)


class UnknownCompanyError(LookupError):
    """Raised when a company does not exist in the current database."""


class UnknownFieldError(LookupError):
    """Raised when a requested field is not available for lookup."""


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    """Hidden attributes for one synthetic company."""

    revenue: int
    profit: int
    employees: int
    growth_rate: float


class SyntheticCompanyDatabase:
    """Generate the same company records whenever the same seed is used."""

    def __init__(self, seed: int = 0) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")

        self._seed = seed
        random_generator = random.Random(seed)
        self._companies = {
            company: self._generate_record(random_generator)
            for company in COMPANY_NAMES
        }

    @staticmethod
    def _generate_record(random_generator: random.Random) -> CompanyRecord:
        revenue = random_generator.randint(100, 5_000)
        profit = random_generator.randint(max(1, revenue // 20), revenue // 3)
        employees = random_generator.randint(50, 50_000)
        growth_rate = round(random_generator.uniform(-0.10, 0.35), 4)
        return CompanyRecord(
            revenue=revenue,
            profit=profit,
            employees=employees,
            growth_rate=growth_rate,
        )

    @property
    def seed(self) -> int:
        """Return the seed used to create this database."""

        return self._seed

    def list_companies(self) -> list[str]:
        """Return company names without exposing their hidden attributes."""

        return list(self._companies)

    def lookup(self, company: str, field: str) -> int | float:
        """Return one allowed attribute for one company."""

        if company not in self._companies:
            raise UnknownCompanyError(f"Unknown company: {company}")
        if field not in SUPPORTED_FIELDS:
            raise UnknownFieldError(f"Unknown field: {field}")

        return getattr(self._companies[company], field)
