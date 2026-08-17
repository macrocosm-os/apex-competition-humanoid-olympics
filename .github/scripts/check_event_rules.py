"""Semantic checks for the deliberately hard Olympic event rules.

These checks exercise the referee-owned state machines directly with synthetic
foot contacts.  They complement a physics rollout: a weak policy rarely
reaches every jump phase, while these assertions catch a future rule change
that accidentally turns a void or wrong-foot touch into a valid landing.
"""

from __future__ import annotations

import math

from env import OlympicsSim, instance_score, instance_spec
from env.course import (HIGH_JUMP_BARS_M, HURDLE_HEIGHT_M, LONG_LANDING_M, LONG_TAKEOFF_M,
                        TRIPLE_LANDING_M, TRIPLE_TAKEOFF_M, build_event)
from env.sim import (HIGH_CLEARANCE_MARGIN_M, HIGH_LANDING_OFFSET_M, MIN_AIRBORNE_STEPS,
                     MIN_SUPPORT_STEPS, PLINTH_TOP, _scene_xml)


def sim(event: str) -> OlympicsSim:
    result = OlympicsSim(instance_spec(event, 0, seed=7))
    result.reset()
    return result


# The public geometry carries the promised hard dimensions, and the decorative
# floor cannot ever become a lower route.
hurdles = build_event("hurdles_100")
assert len([s for s in hurdles.surfaces if s.kind == "hurdle"]) == 10
assert math.isclose(next(s for s in hurdles.surfaces if s.kind == "hurdle").hz * 2,
                    HURDLE_HEIGHT_M)
long_layout = build_event("long_jump")
assert math.isclose(long_layout.challenge["landing_x_m"] - long_layout.challenge["takeoff_x_m"], 6.0)
assert 'contype="0" conaffinity="0"' in _scene_xml(long_layout)

# Ducking under a high-jump bar cannot clear it.  A genuine airborne crossing
# followed by a far-side foot landing can.
high = sim("high_jump")
bar_x = float(high.layout.challenge["bar_x_m"])
bar_h = float(high.layout.challenge["bar_height_m"])
high._hits = lambda kind: False  # type: ignore[method-assign]
high._foot_contacts = lambda: {}  # type: ignore[method-assign]
high._observe_high_jump(bar_x - 0.01, bar_x + 0.01, PLINTH_TOP + bar_h)
assert high._event_reason == "bar_missed", high._event_reason

high = sim("high_jump")
bar_x = float(high.layout.challenge["bar_x_m"])
bar_h = float(high.layout.challenge["bar_height_m"])
high._hits = lambda kind: False  # type: ignore[method-assign]
high._foot_contacts = lambda: {}  # type: ignore[method-assign]
high._high_airborne_steps = MIN_AIRBORNE_STEPS - 1
high._observe_high_jump(bar_x - 0.01, bar_x + 0.01, PLINTH_TOP + bar_h + HIGH_CLEARANCE_MARGIN_M)
assert high._high_valid_crossing, high._event_reason
high._foot_contacts = lambda: {"left": {"track"}}  # type: ignore[method-assign]
high._jump_contact_illegal = lambda contacts, allowed: False  # type: ignore[method-assign]
high._observe_high_jump(bar_x + HIGH_LANDING_OFFSET_M, bar_x + HIGH_LANDING_OFFSET_M + 0.01,
                        PLINTH_TOP + bar_h)
assert high._event_reason == "cleared", high._event_reason

# A long jump only counts after sustained foot support on sand.  A runway/floor
# contact after take-off is a foul, not a shortcut.
long = sim("long_jump")
long._jump_state = "long_flight"
long._observe_long_jump(LONG_LANDING_M, LONG_TAKEOFF_M, {"left": {"track"}})
assert long._event_reason == "jump_foul", long._event_reason
long = sim("long_jump")
long._jump_state = "long_flight"
for _ in range(MIN_SUPPORT_STEPS):
    long._observe_long_jump(LONG_LANDING_M, LONG_TAKEOFF_M, {"left": {"sand"}})
assert long._event_reason == "landed", long._event_reason
assert math.isclose(long._jump_distance, 6.0)

# Triple jump needs same-foot hop, a real second flight, opposite-foot step,
# another flight, then sustained sand support.  A wrong-foot first landing is
# terminally illegal.
triple = sim("triple_jump")
triple._jump_state, triple._takeoff_foot = "hop_flight", "left"
triple._observe_triple_jump(14.1, TRIPLE_TAKEOFF_M, {"right": {"hop_pad"}})
assert triple._event_reason == "jump_foul", triple._event_reason

triple = sim("triple_jump")
triple._jump_state, triple._takeoff_foot = "hop_flight", "left"
for _ in range(MIN_SUPPORT_STEPS):
    triple._observe_triple_jump(14.1, TRIPLE_TAKEOFF_M, {"left": {"hop_pad"}})
assert triple._jump_state == "hop_support" and triple._triple_phase == 1
for _ in range(MIN_AIRBORNE_STEPS):
    triple._observe_triple_jump(15.7, TRIPLE_TAKEOFF_M, {})
assert triple._jump_state == "step_flight"
for _ in range(MIN_SUPPORT_STEPS):
    triple._observe_triple_jump(18.1, TRIPLE_TAKEOFF_M, {"right": {"step_pad"}})
assert triple._jump_state == "step_support" and triple._triple_phase == 2
for _ in range(MIN_AIRBORNE_STEPS):
    triple._observe_triple_jump(19.7, TRIPLE_TAKEOFF_M, {})
assert triple._jump_state == "sand_flight"
for _ in range(MIN_SUPPORT_STEPS):
    triple._observe_triple_jump(TRIPLE_LANDING_M, TRIPLE_TAKEOFF_M, {"left": {"sand"}})
assert triple._event_reason == "landed" and triple._triple_phase == 3
assert math.isclose(triple._jump_distance, TRIPLE_LANDING_M - TRIPLE_TAKEOFF_M)

# Legal completions receive the reserved finish band and the old triple-score
# saturation is gone: a minimum legal landing is not a perfect triple jump.
assert math.isclose(instance_score("long_jump", "landed", 0, 0, 1,
                                   {"jump_distance_m": 6.0}), 0.25)
assert math.isclose(instance_score("triple_jump", "landed", 0, 0, 1,
                                   {"jump_distance_m": 13.0}), 0.25)
assert math.isclose(instance_score("triple_jump", "landed", 0, 0, 1,
                                   {"jump_distance_m": 18.0}), 1.0)
assert math.isclose(instance_score("high_jump", "cleared", 0, 0, 1,
                                   {"bar_height_m": HIGH_JUMP_BARS_M[-1]}), 1.0)

print("ok: hard geometry, anti-duck, legal jumps, and finish scoring")
