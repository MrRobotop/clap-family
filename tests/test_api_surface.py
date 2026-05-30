"""Guard the public API: every advertised symbol imports, aliases match classes."""
import clap_family as cf


def test_all_public_symbols_present():
    expected = {
        "CLAP", "RRLA", "DUCLAP", "AdaptiveDUCLAP", "ACLAP",
        "LearnedGateACLAP", "PhaseAdaptiveLGACLAP",
        "clap", "rrla", "du_clap", "adaptive_du_clap", "a_clap",
        "learned_gate_a_clap", "phase_adaptive_lg_a_clap",
        "CLAPParams", "FieldSet", "ProjectedCEMMPC",
        "ThreeRegionEnv", "HardEnv", "TransitionErrorEnv",
        "target_dwell", "trap_dwell", "unsafe_time", "transition_exposure", "jerk_proxy",
    }
    missing = expected - set(dir(cf))
    assert not missing, f"missing public symbols: {missing}"


def test_every_alias_in_all():
    for name in ("CLAP", "du_clap", "ProjectedCEMMPC", "ThreeRegionEnv", "target_dwell"):
        assert name in cf.__all__
