"""Base validation reports and pipeline composition."""

from abc import ABC, abstractmethod
from typing import Iterable

from pydantic import BaseModel, Field

from src.schema.episode import Episode


class ValidationReport(BaseModel):
    """Explicit validation result with errors and warnings."""

    passed: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Record an error and mark the report as failed."""
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(message)

    def merge(self, other: "ValidationReport") -> None:
        """Merge another report into this report."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.passed = self.passed and other.passed


class Validator(ABC):
    """Base class for episode validators."""

    @abstractmethod
    def validate(self, episode: Episode) -> ValidationReport:
        """Validate one episode."""


class ValidatorPipeline:
    """Run multiple validators and aggregate their reports."""

    def __init__(self, validators: Iterable[Validator]) -> None:
        self.validators = list(validators)

    def validate(self, episode: Episode) -> ValidationReport:
        """Validate one episode with all configured validators."""
        report = ValidationReport()
        for validator in self.validators:
            report.merge(validator.validate(episode))
        return report

    def validate_many(self, episodes: Iterable[Episode]) -> ValidationReport:
        """Validate many episodes with all configured validators."""
        report = ValidationReport()
        for episode in episodes:
            report.merge(self.validate(episode))
        return report
