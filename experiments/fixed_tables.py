from experiments.arm_names import (
    LEARNED_DIRECT,
    TABLE_AVG,
    TABLE_AVG_SWITCHED,
    TABLE_GLOBAL,
    TABLE_RULE,
    TABLE_RULE_SWITCHED,
    TRIPLE_DIRECT,
)
from pipeline.experiment import GRAMMAR, Experiment, against_full

OTHERS = (
    TABLE_AVG,
    TABLE_AVG_SWITCHED,
    TABLE_RULE,
    TABLE_RULE_SWITCHED,
    TABLE_GLOBAL,
    TRIPLE_DIRECT,
)


class FixedTables(Experiment):
    name = "fixed-tables"
    primary = GRAMMAR
    equivalent = False
    pairs = against_full(OTHERS, LEARNED_DIRECT)
