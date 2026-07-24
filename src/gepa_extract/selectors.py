"""Coverage-guaranteed field selection.

GEPA's budget is denominated in *metric calls* -- document extractions -- which
says nothing about how many fields ever get a turn. With 100 fields and a
budget of a few hundred extractions, stock ``round_robin`` reaches perhaps a
dozen of them and the rest keep their seed descriptions, silently.

Worse, ``round_robin``'s cursor is stored **per parent candidate**
(``state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx]``),
and a child inherits ``max()`` of its parents'. As the Pareto front grows, the
population walks several independent cursors and coverage fragments: some
fields are proposed for repeatedly on one lineage while others are never
reached on any.

This module replaces both halves of that with a unit the caller can reason
about -- a **round**, meaning "this field was selected for reflection once":

* :class:`FieldCoverageSelector` keeps one global round count and always serves
  the least-served field that still needs work.
* :class:`RoundsPerFieldStopper` ends the run once every such field has had its
  rounds, delegating the eligibility question back to the selector so there is
  exactly one definition of "needs work".

Together they turn "I have 100 fields, give each of them 2 attempts" into
something stated directly, rather than back-solved from a metric-call budget.

On vocabulary: GEPA calls the unit it evolves a *component*, and this package
calls it a *field* -- the same thing seen from the two sides of the seam, as
``FieldSpec.component_name`` already records. Names here are domain-side
because that is where the module lives; it duck-types GEPA's
``ReflectionComponentSelector`` and ``StopperProtocol`` without importing
either, so ``adapter.py`` and ``optimize.py`` remain the only modules that
import gepa. ``tests/test_gepa_contracts.py`` pins both protocols so the
duck-typing cannot rot unnoticed.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["FieldCoverageSelector", "RoundsPerFieldStopper"]


class FieldCoverageSelector:
    """Selects the least-served field that has not yet been solved.

    Two properties matter, and both are load-bearing for the guarantee:

    *One global counter.* Not per candidate. Which parent the Pareto selector
    happened to pick has no bearing on whose turn it is, so coverage no longer
    fragments as the front grows.

    *Eligibility is monotone.* A field is retired once **any** candidate has
    scored a perfect 1.0 on it, and retirement is permanent. Judging against
    the current parent instead would let a field re-enter after a regression on
    some unrelated lineage, and the stopper's ``min()`` could then never rise --
    a run that cannot terminate. Monotonicity is what makes termination
    provable: every call increments exactly one count, the eligible set only
    shrinks, so the minimum round count is non-decreasing and must reach the
    target.

    Ties break on field path, not at random, so a run is reproducible without
    threading an RNG through.

    Args:
        perfect_score: The score at which a field is considered solved and stops
            consuming turns. Matches gepa's own ``perfect_score`` parameter.
    """

    def __init__(self, *, perfect_score: float = 1.0) -> None:
        self.rounds: Counter[str] = Counter()
        self.perfect_score = perfect_score
        self._retired: set[str] = set()

    def __call__(
        self,
        state: Any,
        trajectories: Any,
        subsample_scores: Any,
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        eligible = self.eligible(state, candidate)
        # Every field solved: fall back to the full set rather than returning
        # []. An empty list would have gepa build an empty reflective dataset
        # and skip, burning the iteration for nothing. The stopper should have
        # ended the run by now; this is the path where a caller supplied their
        # own stop condition instead.
        pool = eligible or sorted(candidate)
        name = min(pool, key=lambda path: (self.rounds[path], path))
        # Counted on *selection*, not on a successful proposal. A field whose
        # reflection keeps returning nothing usable would otherwise be picked
        # forever, and the run would never terminate.
        self.rounds[name] += 1
        return [name]

    def eligible(self, state: Any, candidate: dict[str, str]) -> list[str]:
        """Fields still worth spending a round on, cheapest-first order aside.

        Reads ``state.prog_candidate_objective_scores`` -- the per-field
        aggregate scores gepa maintains for every candidate, which exist because
        the adapter reports ``objective_scores``. Without them (a caller running
        ``frontier_type='instance'``) nothing is ever retired, and this degrades
        to fair round-robin over all fields, which is still an improvement on
        the fragmented cursor.
        """
        for scores in getattr(state, "prog_candidate_objective_scores", None) or []:
            for path, score in scores.items():
                if score >= self.perfect_score:
                    self._retired.add(path)
        return [path for path in sorted(candidate) if path not in self._retired]

    def satisfied(self, state: Any, candidate: dict[str, str], rounds_per_field: int) -> bool:
        """Has every field that still needs work had ``rounds_per_field`` turns?"""
        eligible = self.eligible(state, candidate)
        if not eligible:
            return True
        return min(self.rounds[path] for path in eligible) >= rounds_per_field


class RoundsPerFieldStopper:
    """Stops once every unsolved field has had ``rounds_per_field`` rounds.

    Deliberately counts *rounds*, not iterations. Some iterations never reach
    the selector at all -- gepa skips the whole task when every score
    in the sampled minibatch is already perfect
    (``reflective_mutation.py``, guarded by gepa's ``skip_perfect_score``), and
    an exception mid-iteration is swallowed the same way. Those advance
    ``state.i`` without crediting anyone a turn, so an iteration-denominated
    stop condition delivers "2 rounds each, minus leakage" -- a promise that
    quietly weakens exactly as fields start passing.

    Pair with a ``MaxCandidateProposalsStopper`` backstop: this stopper alone
    puts no ceiling on what a pathological run may spend.
    """

    def __init__(self, selector: FieldCoverageSelector, rounds_per_field: int) -> None:
        if rounds_per_field < 1:
            raise ValueError("rounds_per_field must be at least 1.")
        self.selector = selector
        self.rounds_per_field = rounds_per_field

    def __call__(self, gepa_state: Any) -> bool:
        candidates = getattr(gepa_state, "program_candidates", None)
        if not candidates:
            return False
        # program_candidates[0] is the seed, whose key set is the field set and
        # is fixed for the whole run by construction (see schema.bind).
        return self.selector.satisfied(gepa_state, candidates[0], self.rounds_per_field)
