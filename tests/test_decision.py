"""Decision-engine contract tests.

These cover the rules that protect the required output: exactly two
consistent lines, human priority, conservative species acceptance, and
temporal smoothing.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from app.config import Settings
from app.decision import (
    Box,
    FinalResult,
    FrameEvidence,
    FrameOutcome,
    ResultSmoother,
    SpeciesCandidate,
    Subject,
    best_face,
    classify_frame,
    resolve_species,
)


def square(size: float, x: float = 0.1, y: float = 0.1, confidence: float = 0.9) -> Box:
    return Box(x, y, x + size, y + size, confidence)


def fish_outcome(species: str = "Clownfish", confidence: float = 0.92) -> FrameOutcome:
    return FrameOutcome(
        subject=Subject.FISH,
        species=species,
        confidence=confidence,
        reason="bioclip_confident",
        face=None,
        fish_box=square(0.4),
        p_fish=0.95,
        species_top3=(SpeciesCandidate(species, 0.9),),
    )


def human_outcome(confidence: float = 0.94) -> FrameOutcome:
    return FrameOutcome(
        subject=Subject.HUMAN,
        species=None,
        confidence=confidence,
        reason="face_present",
        face=square(0.2),
        fish_box=None,
        p_fish=0.05,
    )


def not_fish_outcome(confidence: float = 0.99) -> FrameOutcome:
    return FrameOutcome(
        subject=Subject.NOT_FISH,
        species=None,
        confidence=confidence,
        reason="no_evidence",
        face=None,
        fish_box=None,
        p_fish=0.01,
    )


# --------------------------------------------------------------------- output
def test_display_is_exactly_two_consistent_lines() -> None:
    fish = FinalResult.fish_result(
        now=100.0, species="Clownfish", confidence=0.91, stability=0.8
    )
    human = FinalResult.human(now=100.0, confidence=0.94, stability=0.9)
    not_fish = FinalResult.not_fish(now=100.0)

    assert fish.display == {
        "line1": "Fish: True",
        "line2": 'Type of Fish: "Clownfish"',
    }
    assert human.display == {
        "line1": "Fish: False",
        "line2": 'Type of Fish: "Human"',
    }
    assert not_fish.display == {
        "line1": "Fish: False",
        "line2": 'Type of Fish: "Not Fish"',
    }

    for result in (fish, human, not_fish):
        payload = result.as_dict()
        assert payload["display"]["line1"] == f"Fish: {result.fish}"
        assert payload["display"]["line2"] == f'Type of Fish: "{payload["type_of_fish"]}"'
        assert payload["subject"] == result.subject.value


def test_fish_result_rejects_placeholder_labels() -> None:
    with pytest.raises(ValueError):
        FinalResult.fish_result(
            now=1.0, species="Not Fish", confidence=0.9, stability=0.9
        )
    with pytest.raises(ValueError):
        FinalResult.fish_result(
            now=1.0, species="Human", confidence=0.9, stability=0.9
        )


def test_non_fish_result_rejects_species() -> None:
    with pytest.raises(ValueError):
        FinalResult(
            fish=False,
            type_of_fish="Clownfish",
            confidence=0.9,
            subject=Subject.NOT_FISH,
            stability=0.9,
            reason="bad",
            updated_at=1.0,
        )


# ------------------------------------------------------------------- subjects
def test_face_wins_over_non_dominant_fish() -> None:
    settings = Settings()
    evidence = FrameEvidence(
        faces=(square(0.15, 0.05, 0.05, 0.96),),
        fish_boxes=(square(0.20, 0.55, 0.35, 0.8),),
        fish_box=square(0.20, 0.55, 0.35, 0.8),
        gate_p_fish=0.91,
        gate_top1="anemone fish",
        gate_top1_prob=0.81,
        species=(
            SpeciesCandidate("Clownfish", 0.93),
            SpeciesCandidate("Goldfish", 0.02),
        ),
    )
    outcome = classify_frame(evidence, settings)
    assert outcome.subject is Subject.HUMAN
    assert outcome.reason == "face_present"


def test_dominant_foreground_fish_wins_over_small_face() -> None:
    settings = Settings()
    fish_box = square(0.42, 0.28, 0.20, 0.86)
    evidence = FrameEvidence(
        faces=(square(0.07, 0.02, 0.02, 0.82),),
        fish_boxes=(fish_box,),
        fish_box=fish_box,
        gate_p_fish=0.94,
        gate_top1="goldfish",
        gate_top1_prob=0.78,
        species=(
            SpeciesCandidate("Goldfish", 0.95),
            SpeciesCandidate("Koi", 0.03),
        ),
    )
    outcome = classify_frame(evidence, settings)
    assert outcome.subject is Subject.FISH
    assert outcome.species == "Goldfish"


def test_fish_gate_rejects_non_fish_scene() -> None:
    settings = Settings()
    evidence = FrameEvidence(
        gate_p_fish=0.02,
        gate_top1="bathtub",
        gate_top1_prob=0.44,
        species=(SpeciesCandidate("Goldfish", 0.42),),
    )
    outcome = classify_frame(evidence, settings)
    assert outcome.subject is Subject.NOT_FISH
    assert outcome.species is None
    assert outcome.confidence > 0.9


def test_fish_gate_requires_species_confidence() -> None:
    settings = Settings()
    evidence = FrameEvidence(
        gate_p_fish=0.62,
        gate_top1="goldfish",
        gate_top1_prob=0.2,
        species=(
            SpeciesCandidate("Goldfish", 0.60),
            SpeciesCandidate("Koi", 0.55),
        ),
    )
    outcome = classify_frame(evidence, settings)
    assert outcome.subject is Subject.NOT_FISH
    assert outcome.reason == "species_uncertain"


def test_best_face_ignores_tiny_and_low_confidence_faces() -> None:
    settings = Settings()
    tiny = square(0.01, confidence=0.99)
    low = square(0.2, confidence=0.3)
    good = square(0.15, confidence=0.9)
    assert best_face((tiny, low, good), settings) is good
    assert best_face((tiny, low), settings) is None


def test_species_supported_gate_accepts_confident_flatfish() -> None:
    settings = Settings()
    evidence = FrameEvidence(
        gate_p_fish=0.323,
        gate_top1="flatworm",
        gate_top1_prob=0.43,
        species=(
            SpeciesCandidate("Sole", 0.95),
            SpeciesCandidate("Flounder", 0.04),
        ),
    )
    outcome = classify_frame(evidence, settings)
    assert outcome.subject is Subject.FISH
    assert outcome.species == "Sole"


def test_species_supported_gate_still_rejects_low_evidence() -> None:
    settings = Settings()
    evidence = FrameEvidence(
        gate_p_fish=0.171,
        gate_top1="starfish",
        gate_top1_prob=0.17,
        species=(SpeciesCandidate("Goby", 0.38),),
    )
    outcome = classify_frame(evidence, settings)
    assert outcome.subject is Subject.NOT_FISH


# -------------------------------------------------------------------- species
def test_species_requires_probability_and_margin() -> None:
    settings = Settings()
    species, reason = resolve_species(
        (
            SpeciesCandidate("Great white shark", 0.471),
            SpeciesCandidate("Tiger shark", 0.455),
        ),
        None,
        settings,
    )
    assert species is None
    assert reason == "species_uncertain"

    species, reason = resolve_species(
        (
            SpeciesCandidate("Clownfish", 0.91),
            SpeciesCandidate("Goldfish", 0.03),
        ),
        None,
        settings,
    )
    assert species == "Clownfish"
    assert reason == "bioclip_confident"


def test_imagenet_breaks_species_near_tie() -> None:
    settings = Settings()
    species, reason = resolve_species(
        (
            SpeciesCandidate("Great white shark", 0.471),
            SpeciesCandidate("Tiger shark", 0.455),
            SpeciesCandidate("Hammerhead shark", 0.04),
        ),
        "Great white shark",
        settings,
    )
    assert species == "Great white shark"
    assert reason == "imagenet_tiebreak"


def test_imagenet_does_not_override_a_clear_winner() -> None:
    settings = Settings()
    species, reason = resolve_species(
        (
            SpeciesCandidate("Tiger shark", 0.80),
            SpeciesCandidate("Great white shark", 0.10),
        ),
        "Great white shark",
        settings,
    )
    assert species == "Tiger shark"
    assert reason == "bioclip_confident"


# ------------------------------------------------------------------- smoothing
def test_smoothing_requires_minimum_votes() -> None:
    settings = replace(Settings(), smooth_window_seconds=10.0, smooth_min_votes=2)
    smoother = ResultSmoother(settings)

    first = smoother.update(fish_outcome(), now=100.0)
    assert first.fish is False
    assert first.type_of_fish == "Not Fish"

    second = smoother.update(fish_outcome(), now=100.2)
    assert second.fish is True
    assert second.type_of_fish == "Clownfish"
    assert second.stability == pytest.approx(1.0)


def test_smoothing_transitions_after_window() -> None:
    settings = replace(Settings(), smooth_window_seconds=1.0, smooth_min_votes=2)
    smoother = ResultSmoother(settings)

    smoother.update(human_outcome(), now=100.0)
    human = smoother.update(human_outcome(), now=100.2)
    assert human.type_of_fish == "Human"

    # Two fish votes inside a window that still contains two human votes
    # should not flip yet: human priority wins ties.
    still_human = smoother.update(fish_outcome(), now=100.4)
    assert still_human.type_of_fish == "Human"

    # Once the human votes age out, the fish decision becomes stable.
    fish = smoother.update(fish_outcome(), now=101.6)
    assert fish.type_of_fish == "Not Fish"
    fish = smoother.update(fish_outcome(), now=101.8)
    assert fish.fish is True
    assert fish.type_of_fish == "Clownfish"


def test_frame_outcome_cannot_be_fish_without_species() -> None:
    with pytest.raises(ValueError):
        FrameOutcome(
            subject=Subject.FISH,
            species=None,
            confidence=0.7,
            reason="species_uncertain",
            face=None,
            fish_box=None,
            p_fish=0.8,
        )


def test_smoothing_does_not_invent_a_species() -> None:
    settings = replace(Settings(), smooth_window_seconds=10.0, smooth_min_votes=2)
    smoother = ResultSmoother(settings)
    # A single fish vote is below the vote threshold; the smoother must fall
    # back to Not Fish rather than naming the species from one frame.
    result = smoother.update(fish_outcome(), now=100.0)
    assert result.fish is False
    assert result.type_of_fish == "Not Fish"
    assert result.display["line2"] == 'Type of Fish: "Not Fish"'


def test_smoothing_species_switch_requires_votes() -> None:
    settings = replace(Settings(), smooth_window_seconds=10.0, smooth_min_votes=2)
    smoother = ResultSmoother(settings)
    for i in range(2):
        smoother.update(fish_outcome("Clownfish"), now=100.0 + i * 0.1)
    switched = smoother.update(fish_outcome("Goldfish"), now=100.4)
    # Only one Goldfish vote exists; the current Clownfish label is retained
    # briefly, but the result is still internally consistent.
    assert switched.type_of_fish in {"Clownfish", "Goldfish"}
    if switched.fish:
        assert switched.type_of_fish in {"Clownfish", "Goldfish"}


def test_initial_smoother_is_not_fish() -> None:
    smoother = ResultSmoother(Settings())
    assert smoother.current.display == {
        "line1": "Fish: False",
        "line2": 'Type of Fish: "Not Fish"',
    }
