"""Frame-level evidence fusion and temporal smoothing.

The required output is always produced here, from one atomic result object, so
the UI can never render a contradictory pair such as
``Fish: True`` / ``Type of Fish: "Not Fish"``.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .config import Settings
from .species import IMAGENET_FISH_CLASSES, SPECIES_BY_NAME


class Subject(str, Enum):
    FISH = "fish"
    HUMAN = "human"
    NOT_FISH = "not_fish"


@dataclass(frozen=True)
class Box:
    """Normalized box (0..1) plus detector confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamped(self) -> "Box":
        return Box(
            min(max(self.x1, 0.0), 1.0),
            min(max(self.y1, 0.0), 1.0),
            min(max(self.x2, 0.0), 1.0),
            min(max(self.y2, 0.0), 1.0),
            self.confidence,
        )

    def as_dict(self) -> dict:
        return {
            "x1": round(self.x1, 5),
            "y1": round(self.y1, 5),
            "x2": round(self.x2, 5),
            "y2": round(self.y2, 5),
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class SpeciesCandidate:
    name: str
    probability: float

    def as_dict(self) -> dict:
        return {"name": self.name, "probability": round(self.probability, 4)}


@dataclass(frozen=True)
class FrameEvidence:
    """Everything the models observed for one analyzed frame."""

    faces: tuple[Box, ...] = ()
    fish_boxes: tuple[Box, ...] = ()
    fish_box: Box | None = None
    gate_p_fish: float = 0.0
    gate_top1: str = ""
    gate_top1_prob: float = 0.0
    species: tuple[SpeciesCandidate, ...] = ()
    species_tie: str | None = None
    inference_ms: float = 0.0


@dataclass(frozen=True)
class FrameOutcome:
    subject: Subject
    species: str | None
    confidence: float
    reason: str
    face: Box | None
    fish_box: Box | None
    p_fish: float
    species_top3: tuple[SpeciesCandidate, ...] = ()

    def __post_init__(self) -> None:
        if self.subject is Subject.FISH and not self.species:
            raise ValueError("A fish outcome must carry a species")


@dataclass(frozen=True)
class FinalResult:
    fish: bool
    type_of_fish: str
    confidence: float
    subject: Subject
    stability: float
    reason: str
    updated_at: float
    face: Box | None = None
    fish_box: Box | None = None
    p_fish: float = 0.0
    species_top3: tuple[SpeciesCandidate, ...] = ()
    stale: bool = False

    def __post_init__(self) -> None:
        if self.fish:
            if self.type_of_fish in {"Not Fish", "Human"}:
                raise ValueError("A fish result cannot use a non-fish label")
            if self.type_of_fish not in SPECIES_BY_NAME:
                raise ValueError(f"Unsupported fish label: {self.type_of_fish!r}")
        else:
            if self.type_of_fish not in {"Not Fish", "Human"}:
                raise ValueError("A non-fish result must be Not Fish or Human")

    @property
    def display(self) -> dict[str, str]:
        return {
            "line1": f"Fish: {self.fish}",
            "line2": f'Type of Fish: "{self.type_of_fish}"',
        }

    def as_dict(self) -> dict:
        return {
            "fish": self.fish,
            "type_of_fish": self.type_of_fish,
            "display": self.display,
            "confidence": round(self.confidence, 4),
            "subject": self.subject.value,
            "stability": round(self.stability, 3),
            "reason": self.reason,
            "updated_at": self.updated_at,
            "face": self.face.as_dict() if self.face else None,
            "fish_box": self.fish_box.as_dict() if self.fish_box else None,
            "p_fish": round(self.p_fish, 4),
            "species_top3": [c.as_dict() for c in self.species_top3],
            "stale": self.stale,
        }

    @classmethod
    def not_fish(
        cls,
        now: float,
        reason: str = "no_evidence",
        confidence: float = 0.0,
        stability: float = 0.0,
        **kwargs,
    ) -> "FinalResult":
        return cls(
            False,
            "Not Fish",
            confidence,
            Subject.NOT_FISH,
            stability,
            reason,
            now,
            **kwargs,
        )

    @classmethod
    def human(cls, now: float, confidence: float, stability: float, reason: str = "face_present", **kwargs) -> "FinalResult":
        return cls(False, "Human", confidence, Subject.HUMAN, stability, reason, now, **kwargs)

    @classmethod
    def fish_result(
        cls,
        now: float,
        species: str,
        confidence: float,
        stability: float,
        reason: str = "fish_confident",
        **kwargs,
    ) -> "FinalResult":
        return cls(True, species, confidence, Subject.FISH, stability, reason, now, **kwargs)


def best_face(faces: Iterable[Box], settings: Settings) -> Box | None:
    candidates = [
        f
        for f in faces
        if f.confidence >= settings.face_min_confidence
        and settings.face_min_area <= f.area <= settings.face_max_area
    ]
    return max(candidates, key=lambda b: b.area, default=None)


def best_fish_box(boxes: Iterable[Box], settings: Settings) -> Box | None:
    candidates = [
        b
        for b in boxes
        if b.confidence >= settings.yolo_fish_confidence
        and b.area >= settings.yolo_min_box_area
    ]
    if not candidates:
        return None
    # Prefer a confident, reasonably large box; pure confidence can favor a
    # tiny false positive in the background.
    return max(candidates, key=lambda b: b.confidence * (b.area ** 0.20))


def resolve_species(
    candidates: Iterable[SpeciesCandidate],
    tie: str | None,
    settings: Settings,
) -> tuple[str | None, str]:
    """Return (species, reason) only when the species is trustworthy."""
    ranked = tuple(sorted(candidates, key=lambda c: c.probability, reverse=True))
    if not ranked:
        return None, "no_species_prediction"

    top = ranked[0]
    second = ranked[1].probability if len(ranked) > 1 else 0.0
    margin = top.probability - second

    # A confident ImageNet class can resolve a BioCLIP near-tie (for example
    # Great White Shark vs Tiger Shark).
    if tie and margin < settings.species_min_margin:
        for candidate in ranked[:3]:
            if candidate.name == tie and top.probability >= settings.species_tie_min_p:
                return tie, "imagenet_tiebreak"

    if top.probability >= settings.species_min_p and margin >= settings.species_min_margin:
        return top.name, "bioclip_confident"

    if tie and top.name == tie and top.probability >= settings.species_tie_min_p:
        return tie, "imagenet_tiebreak"

    return None, "species_uncertain"


def classify_frame(evidence: FrameEvidence, settings: Settings) -> FrameOutcome:
    """Apply the primary-subject and conservative fish rules to one frame."""
    face = best_face(evidence.faces, settings)
    fish_box = best_fish_box(evidence.fish_boxes, settings)
    if evidence.fish_box is not None:
        fish_box = evidence.fish_box

    species, species_reason = resolve_species(evidence.species, evidence.species_tie, settings)

    top1_is_fish = evidence.gate_top1 in IMAGENET_FISH_CLASSES
    gate_ok = evidence.gate_p_fish >= settings.fish_gate_p and (
        (top1_is_fish and evidence.gate_top1_prob >= settings.fish_gate_top1)
        or evidence.gate_p_fish >= 0.45
        # Flatfish and similar species can sit below the strong gate while
        # still having an unambiguous BioCLIP match (for example sole).  A
        # resolved species plus a moderate fish probability is accepted, but
        # the non-fish scenes observed during calibration stay below 0.28.
        or (
            species is not None
            and evidence.gate_p_fish >= settings.fish_gate_species_p
        )
    )
    fish_valid = gate_ok and species is not None

    if face is not None:
        # A clearly dominant fish (foreground object) wins over a small face
        # in the background.  Otherwise a detected face is always "Human".
        dominant = False
        if fish_valid:
            if fish_box is not None:
                dominant = fish_box.area >= max(
                    settings.fish_dominance_min_area,
                    settings.fish_dominance_ratio * face.area,
                )
            else:
                dominant = evidence.gate_p_fish >= settings.fish_dominance_no_box_p
        if not dominant:
            return FrameOutcome(
                subject=Subject.HUMAN,
                species=None,
                confidence=face.confidence,
                reason="face_present",
                face=face,
                fish_box=fish_box if fish_valid else None,
                p_fish=evidence.gate_p_fish,
                species_top3=evidence.species[:3],
            )

    if fish_valid:
        species_p = next((c.probability for c in evidence.species if c.name == species), 0.0)
        box_conf = fish_box.confidence if fish_box else 0.0
        confidence = 0.65 * species_p + 0.25 * evidence.gate_p_fish + 0.10 * box_conf
        return FrameOutcome(
            subject=Subject.FISH,
            species=species,
            confidence=min(1.0, confidence),
            reason=species_reason,
            face=face,
            fish_box=fish_box,
            p_fish=evidence.gate_p_fish,
            species_top3=evidence.species[:3],
        )

    reason = "no_evidence"
    if evidence.gate_p_fish >= settings.fish_gate_p and species is None:
        reason = "species_uncertain"
    elif evidence.gate_p_fish >= settings.fish_gate_p:
        reason = "verifier_rejected"
    elif evidence.fish_boxes:
        reason = "low_fish_evidence"

    return FrameOutcome(
        subject=Subject.NOT_FISH,
        species=None,
        confidence=max(0.0, 1.0 - evidence.gate_p_fish),
        reason=reason,
        face=face,
        fish_box=None,
        p_fish=evidence.gate_p_fish,
        species_top3=evidence.species[:3],
    )


class ResultSmoother:
    """Majority vote with hysteresis over a short time window."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._window: deque[tuple[float, FrameOutcome]] = deque()
        self._current = FinalResult.not_fish(time.time(), reason="initializing")
        self._last_change = time.time()

    @property
    def current(self) -> FinalResult:
        return self._current

    def reset(self, now: float | None = None, reason: str = "camera_idle") -> FinalResult:
        now = time.time() if now is None else now
        self._window.clear()
        self._current = FinalResult.not_fish(now, reason=reason)
        self._last_change = now
        return self._current

    def _prune(self, now: float) -> None:
        cutoff = now - self.settings.smooth_window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        while len(self._window) > 12:
            self._window.popleft()

    def update(self, outcome: FrameOutcome, now: float | None = None) -> FinalResult:
        now = time.time() if now is None else now
        self._window.append((now, outcome))
        self._prune(now)
        recent = [o for _, o in self._window]
        counts = Counter(o.subject for o in recent)
        human_count = counts[Subject.HUMAN]
        fish_count = counts[Subject.FISH]
        not_count = counts[Subject.NOT_FISH]
        min_votes = self.settings.smooth_min_votes

        subject: Subject
        if human_count >= min_votes and human_count >= fish_count:
            subject = Subject.HUMAN
        elif fish_count >= min_votes and fish_count > human_count:
            subject = Subject.FISH
        elif not_count >= min_votes and not_count >= fish_count and not_count >= human_count:
            subject = Subject.NOT_FISH
        elif (
            self._current.subject is not Subject.NOT_FISH
            and now - self._last_change < self.settings.keep_previous_seconds
        ):
            subject = self._current.subject
        else:
            subject = Subject.NOT_FISH

        species: str | None = None
        confidence = 0.0
        stability = 0.0
        reason = outcome.reason
        matching: list[FrameOutcome] = []

        if subject is Subject.HUMAN:
            matching = [o for o in recent if o.subject is Subject.HUMAN]
            confidence = sum(o.confidence for o in matching) / len(matching)
            stability = human_count / len(recent)
            reason = "face_present"
            if self._current.subject is not Subject.HUMAN:
                self._last_change = now
            result = FinalResult.human(
                now,
                confidence=confidence,
                stability=stability,
                reason=reason,
                face=next((o.face for o in reversed(matching) if o.face), None),
                fish_box=next((o.fish_box for o in reversed(matching) if o.fish_box), None),
                p_fish=max((o.p_fish for o in matching), default=0.0),
                species_top3=matching[-1].species_top3 if matching else (),
            )
            self._current = result
            return result

        if subject is Subject.FISH:
            fish_outcomes = [o for o in recent if o.subject is Subject.FISH and o.species]
            species_counts = Counter(o.species for o in fish_outcomes if o.species)
            if species_counts:
                candidate, votes = species_counts.most_common(1)[0]
                if votes >= min_votes:
                    species = candidate
                elif self._current.fish and now - self._last_change < self.settings.keep_previous_seconds:
                    species = self._current.type_of_fish
                else:
                    species = None

            if species is None:
                subject = Subject.NOT_FISH
            else:
                matching = [o for o in fish_outcomes if o.species == species]
                confidence = sum(o.confidence for o in matching) / len(matching)
                stability = votes / len(recent)
                reason = next((o.reason for o in reversed(matching)), "fish_confident")
                if self._current.subject is not Subject.FISH or self._current.type_of_fish != species:
                    self._last_change = now
                result = FinalResult.fish_result(
                    now,
                    species=species,
                    confidence=confidence,
                    stability=stability,
                    reason=reason,
                    face=next((o.face for o in reversed(matching) if o.face), None),
                    fish_box=next((o.fish_box for o in reversed(matching) if o.fish_box), None),
                    p_fish=max((o.p_fish for o in matching), default=0.0),
                    species_top3=matching[-1].species_top3 if matching else (),
                )
                self._current = result
                return result

        if subject is Subject.NOT_FISH:
            not_outcomes = [o for o in recent if o.subject is Subject.NOT_FISH]
            if not_outcomes:
                confidence = sum(o.confidence for o in not_outcomes) / len(not_outcomes)
            stability = not_count / len(recent) if recent else 0.0
            if self._current.subject is not Subject.NOT_FISH:
                self._last_change = now
            result = FinalResult.not_fish(
                now,
                reason=outcome.reason if outcome.subject is Subject.NOT_FISH else "transitioning",
                confidence=confidence,
                stability=stability,
                face=next((o.face for o in reversed(not_outcomes) if o.face), None),
                p_fish=max((o.p_fish for o in not_outcomes), default=0.0),
                species_top3=not_outcomes[-1].species_top3 if not_outcomes else (),
            )
            self._current = result
            return result

        # Defensive fallback: never leave a contradictory state.
        self._current = FinalResult.not_fish(now, reason="fallback")
        return self._current
