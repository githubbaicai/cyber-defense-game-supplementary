

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from common.config import (
    MACOMP_FE_BUDGETS,
    MACOMP_RUNS,
    MACOMP_WORKERS,
    MACOMP_DIR,
    TERMINATION_HV_REFERENCE_POINT,
    TERMINATION_LOSS_SCALE,
)


OUTPUT_DIR = MACOMP_DIR
WORKERS = MACOMP_WORKERS


DEFAULT_FE_BUDGETS = MACOMP_FE_BUDGETS
DEFAULT_RUNS = MACOMP_RUNS
CONFIDENCE_LEVEL = 0.95

BASE_PROBABILITY = 0.1
BASE_ROUNDS = 3



UNLIMITED_RATIO_CODE = 0.0



FACTOR_ANCHOR_RATIO = 0.2


@dataclass(frozen=True)
class Variant:
    

    name: str
    selection_mode: str
    neighborhood: str
    local_search_prob: float
    max_local_search_iters: int
    no_improve_rounds: int = 2
    elite_size: int | None = None

    
    use_archive: bool = False
    archive_max_size: int | None = None

    
    use_greedy_operators: bool = False
    greedy_max_steps: int = 200

    
    use_ma_specific_init: bool = False

    max_local_fe_ratio: float | None = None
    max_local_fe_per_gen: int | None = None

    local_search_trigger: str = "every_gen"
    local_search_start_ratio: float = 0.5
    hv_stagnation_generations: int = 5
    front_stagnation_generations: int = 3
    trigger_hv_tolerance: float = 1e-6

    acceptance_mode: str = "current"
    acceptance_epsilon: float = 1e-3

    trigger_loss_scale: float = TERMINATION_LOSS_SCALE
    trigger_hv_reference_point: tuple[float, float] = (
        TERMINATION_HV_REFERENCE_POINT
    )

    
    last_gen_local_fe_ratio: float = 0.2

    code: str = ""
    group: str = "A"

    def ma_parameters(self) -> dict:
        
        parameters = asdict(self)
        for name in ("name", "code", "group"):
            parameters.pop(name)

        if parameters["selection_mode"] == "none":
            parameters["selection_mode"] = "elite"
        if parameters["neighborhood"] == "none":
            parameters["neighborhood"] = "all"

        return parameters


BASE_VARIANT = Variant(
    name="Elite-ARS",
    selection_mode="elite",
    neighborhood="all",
    local_search_prob=BASE_PROBABILITY,
    max_local_search_iters=BASE_ROUNDS,
    code="A0",
    group="A",
)

CAPPED_BASE = replace(
    BASE_VARIANT,
    max_local_fe_ratio=FACTOR_ANCHOR_RATIO,
)

BASELINE = Variant(
    name="NSGA-II",
    selection_mode="none",
    neighborhood="none",
    local_search_prob=0.0,
    max_local_search_iters=BASE_ROUNDS,
    code="G0",
    group="G",
)

CAP_VARIANTS = tuple(
    replace(
        BASE_VARIANT,
        name=f"A{index}-Cap{int(ratio * 100)}",
        max_local_fe_ratio=ratio,
        code=f"A{index}",
        group="A",
    )
    for index, ratio in enumerate((0.1, 0.2, 0.3, 0.5), start=1)
)

QUOTA_VARIANTS = tuple(
    replace(
        CAPPED_BASE,
        name=f"B{index}-Quota{quota}",
        elite_size=quota,
        code=f"B{index}",
        group="B",
    )
    for index, quota in enumerate((10, 5, 2, 1, 20))
)

NEIGHBORHOOD_VARIANTS = tuple(
    replace(
        CAPPED_BASE,
        name=f"C{index}-{label}",
        neighborhood=neighborhood,
        code=f"C{index}",
        group="C",
    )
    for index, (label, neighborhood) in enumerate(
        (
            ("Swap", "swap"),
            ("AddRemove", "add_remove"),
            ("AddSwap", "add+swap"),
            ("RemoveSwap", "remove+swap"),
        ),
        start=1,
    )
)

DEPTH_VARIANTS = tuple(
    replace(
        CAPPED_BASE,
        name=f"D{index}-Iters{iterations}",
        max_local_search_iters=iterations,
        no_improve_rounds=no_improve,
        code=f"D{index}",
        group="D",
    )
    for index, (iterations, no_improve) in enumerate(
        ((1, 1), (2, 1), (5, 2), (20, 2)),
        start=1,
    )
)

TRIGGER_VARIANTS = tuple(
    replace(
        CAPPED_BASE,
        name=f"E{index}-{label}",
        local_search_trigger=trigger,
        code=f"E{index}",
        group="E",
    )
    for index, (label, trigger) in enumerate(
        (
            ("LastGen", "last_gen"),
            ("AfterHalf", "after_ratio"),
            ("HVStagnation", "hv_stagnation"),
            ("FrontStagnation", "front_stagnation"),
        ),
        start=1,
    )
)

ACCEPTANCE_VARIANTS = tuple(
    replace(
        CAPPED_BASE,
        name=f"F{index}-{label}",
        acceptance_mode=mode,
        code=f"F{index}",
        group="F",
    )
    for index, (label, mode) in enumerate(
        (
            ("Dominance", "dominance"),
            ("WeightedSum", "weighted_sum"),
            ("FirstImprovement", "first_improvement"),
            ("Epsilon", "epsilon"),
        ),
        start=1,
    )
)

COMBINATION_VARIANTS = (
    replace(
        CAPPED_BASE,
        name="G1-Cap20-Quota2-Iters1",
        elite_size=2,
        max_local_search_iters=1,
        no_improve_rounds=1,
        code="G1",
        group="G",
    ),
    replace(
        CAPPED_BASE,
        name="G2-Cap20-Swap-Iters1",
        neighborhood="swap",
        max_local_search_iters=1,
        no_improve_rounds=1,
        code="G2",
        group="G",
    ),
    replace(
        CAPPED_BASE,
        name="G3-Cap20-AfterHalf-Dominance",
        local_search_trigger="after_ratio",
        acceptance_mode="dominance",
        code="G3",
        group="G",
    ),
    replace(
        CAPPED_BASE,
        name="G4-Hybrid",
        elite_size=2,
        neighborhood="swap",
        max_local_search_iters=1,
        no_improve_rounds=1,
        local_search_trigger="after_ratio",
        acceptance_mode="dominance",
        code="G4",
        group="G",
    ),
    replace(
        CAPPED_BASE,
        name="G5-Balanced",
        elite_size=5,
        max_local_search_iters=2,
        no_improve_rounds=1,
        max_local_fe_per_gen=25,
        local_search_trigger="after_ratio",
        acceptance_mode="weighted_sum",
        code="G5",
        group="G",
    ),
)

SELECTION_VARIANTS = (
    replace(
        CAPPED_BASE,
        name="Random-ARS",
        selection_mode="random",
        code="H1",
        group="S",
    ),
)


STRUCTURED_VARIANTS = (
    replace(
        CAPPED_BASE,
        name="S1-Structured",
        neighborhood="structured",
        code="S1",
        group="S",
    ),
)

S2_VARIANT = replace(
    CAPPED_BASE,
    name="S2-Structured-Archive",
    neighborhood="structured",
    use_archive=True,
    code="S2",
    group="S",
)

S3_VARIANT = replace(
    CAPPED_BASE,
    name="S3-Structured-Greedy",
    neighborhood="structured",
    use_greedy_operators=True,
    code="S3",
    group="S",
)

S4_VARIANT = replace(
    CAPPED_BASE,
    name="S4-Structured-Archive-Greedy",
    neighborhood="structured",
    use_archive=True,
    use_greedy_operators=True,
    code="S4",
    group="S",
)


S6_VARIANT = replace(
    CAPPED_BASE,
    name="S6-Structured-PerGen",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    code="S6",
    group="S",
)

S7_VARIANT = replace(
    CAPPED_BASE,
    name="S7-Structured-PerGen-AfterHalf",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    local_search_trigger="after_ratio",
    local_search_start_ratio=0.5,
    code="S7",
    group="S",
)

S8_VARIANT = replace(
    CAPPED_BASE,
    name="S8-Structured-PerGen-AfterHalf-Greedy",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    local_search_trigger="after_ratio",
    local_search_start_ratio=0.5,
    use_greedy_operators=True,
    code="S8",
    group="S",
)

S9_VARIANT = replace(
    CAPPED_BASE,
    name="S9-Full-PerGen",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    local_search_trigger="after_ratio",
    local_search_start_ratio=0.5,
    use_greedy_operators=True,
    use_archive=True,
    code="S9",
    group="S",
)

S10_VARIANT = replace(
    CAPPED_BASE,
    name="S10-Structured-HVStagnation",
    neighborhood="structured",
    local_search_trigger="hv_stagnation",
    hv_stagnation_generations=3,
    code="S10",
    group="S",
)


R1_VARIANT = replace(
    CAPPED_BASE,
    name="R1-PerGen-Only",
    neighborhood="all",
    max_local_fe_per_gen=25,
    code="R1",
    group="R",
)

R2_VARIANT = replace(
    CAPPED_BASE,
    name="R2-PerGen-Structured",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    code="R2",
    group="R",
)

R3_VARIANT = replace(
    CAPPED_BASE,
    name="R3-PerGen-Structured-Greedy",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    use_greedy_operators=True,
    code="R3",
    group="R",
)

R4_VARIANT = replace(
    CAPPED_BASE,
    name="R4-Full",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    use_greedy_operators=True,
    use_archive=True,
    code="R4",
    group="R",
)

R5_VARIANT = replace(
    CAPPED_BASE,
    name="R5-Full-AfterHalf",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    use_greedy_operators=True,
    use_archive=True,
    local_search_trigger="after_ratio",
    local_search_start_ratio=0.5,
    code="R5",
    group="R",
)

R6_VARIANT = replace(
    CAPPED_BASE,
    name="R6-Full-Init",
    neighborhood="structured",
    max_local_fe_per_gen=25,
    use_greedy_operators=True,
    use_archive=True,
    use_ma_specific_init=True,
    code="R6",
    group="R",
)

VARIANTS = (
    BASELINE,
    BASE_VARIANT,
    *CAP_VARIANTS,
    *QUOTA_VARIANTS,
    *NEIGHBORHOOD_VARIANTS,
    *DEPTH_VARIANTS,
    *TRIGGER_VARIANTS,
    *ACCEPTANCE_VARIANTS,
    *COMBINATION_VARIANTS,
    *SELECTION_VARIANTS,
    *STRUCTURED_VARIANTS,
    S2_VARIANT,
    S3_VARIANT,
    S4_VARIANT,
    S6_VARIANT,
    S7_VARIANT,
    S8_VARIANT,
    S9_VARIANT,
    S10_VARIANT,
    R1_VARIANT,
    R2_VARIANT,
    R3_VARIANT,
    R4_VARIANT,
    R5_VARIANT,
    R6_VARIANT,
)

VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
