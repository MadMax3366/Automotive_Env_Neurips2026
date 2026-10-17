"""Registry mapping agent-facing query names to VHAL/sensor backends."""

from automotive_env.sensors.vhal import get_vhal
from automotive_env.sensors.sensor import get_sensor

# query_name -> {"type": "vhal"|"sensor", ...}
QUERY_REGISTRY = {
    # === Vehicle motion (VHAL) ===
    "speed":             {"type": "vhal", "name": "PERF_VEHICLE_SPEED",      "area": 0,    "unit": "km/h"},
    "speed_display":     {"type": "vhal", "name": "PERF_VEHICLE_SPEED_DISPLAY","area": 0,  "unit": "km/h"},
    "gear":              {"type": "vhal", "name": "GEAR_SELECTION",           "area": 0,    "unit": "enum"},
    "current_gear":      {"type": "vhal", "name": "CURRENT_GEAR",             "area": 0,    "unit": "enum"},
    "rpm":               {"type": "vhal", "name": "ENGINE_RPM",               "area": 0,    "unit": "rpm"},
    "steering_angle":    {"type": "vhal", "name": "PERF_STEERING_ANGLE",      "area": 0,    "unit": "deg"},
    "odometer":          {"type": "vhal", "name": "PERF_ODOMETER",            "area": 0,    "unit": "km"},
    "turn_signal":       {"type": "vhal", "name": "TURN_SIGNAL_STATE",        "area": 0,    "unit": "enum"},

    # === Powertrain (VHAL) ===
    "fuel_level":        {"type": "vhal", "name": "FUEL_LEVEL",               "area": 0,    "unit": "%"},
    "fuel_low":          {"type": "vhal", "name": "FUEL_LEVEL_LOW",           "area": 0,    "unit": "bool"},
    "range_remaining":   {"type": "vhal", "name": "RANGE_REMAINING",          "area": 0,    "unit": "m"},
    "ev_battery":        {"type": "vhal", "name": "EV_BATTERY_LEVEL",         "area": 0,    "unit": "%"},
    "engine_coolant":    {"type": "vhal", "name": "ENGINE_COOLANT_TEMP",      "area": 0,    "unit": "°C"},
    "engine_oil_temp":   {"type": "vhal", "name": "ENGINE_OIL_TEMP",          "area": 0,    "unit": "°C"},
    "ignition":          {"type": "vhal", "name": "IGNITION_STATE",           "area": 0,    "unit": "enum"},

    # === Safety state (VHAL) ===
    "parking_brake":     {"type": "vhal", "name": "PARKING_BRAKE_ON",         "area": 0,    "unit": "bool"},
    "abs_active":        {"type": "vhal", "name": "ABS_ACTIVE",               "area": 0,    "unit": "bool"},
    "traction_control":  {"type": "vhal", "name": "TRACTION_CONTROL_ACTIVE",  "area": 0,    "unit": "bool"},
    "night_mode":        {"type": "vhal", "name": "NIGHT_MODE",               "area": 0,    "unit": "bool"},
    "tire_pressure_fl":  {"type": "vhal", "name": "TIRE_PRESSURE",            "area": 1,    "unit": "kPa"},
    "tire_pressure_fr":  {"type": "vhal", "name": "TIRE_PRESSURE",            "area": 2,    "unit": "kPa"},
    "tire_pressure_rl":  {"type": "vhal", "name": "TIRE_PRESSURE",            "area": 4,    "unit": "kPa"},
    "tire_pressure_rr":  {"type": "vhal", "name": "TIRE_PRESSURE",            "area": 8,    "unit": "kPa"},
    "tire_pressure_low_fl": {"type": "vhal", "name": "CRITICALLY_LOW_TIRE_PRESSURE", "area": 1, "unit": "bool"},

    # === HVAC current state (VHAL) ===
    "hvac_on":           {"type": "vhal", "name": "HVAC_POWER_ON",            "area": 0x75, "unit": "bool"},
    "hvac_temp_driver":  {"type": "vhal", "name": "HVAC_TEMPERATURE_SET",     "area": 0x31, "unit": "°C"},
    "hvac_temp_passenger":{"type":"vhal", "name": "HVAC_TEMPERATURE_SET",     "area": 0x44, "unit": "°C"},
    "hvac_fan_speed":    {"type": "vhal", "name": "HVAC_FAN_SPEED",           "area": 0x75, "unit": "level (OFF=1, MAX=6)"},
    "hvac_ac_on":        {"type": "vhal", "name": "HVAC_AC_ON",               "area": 0x75, "unit": "bool"},
    "hvac_recirc":       {"type": "vhal", "name": "HVAC_RECIRC_ON",           "area": 0x75, "unit": "bool"},
    "hvac_auto":         {"type": "vhal", "name": "HVAC_AUTO_ON",             "area": 0x75, "unit": "bool"},
    "hvac_defroster_front":{"type":"vhal","name": "HVAC_DEFROSTER",           "area": 0x1,  "unit": "bool"},
    "hvac_defroster_rear":{"type":"vhal", "name": "HVAC_DEFROSTER",           "area": 0x2,  "unit": "bool"},
    "hvac_seat_heat_driver":{"type":"vhal","name":"HVAC_SEAT_TEMPERATURE",    "area": 0x1,  "unit": "level (0=OFF, 2=MAX)"},

    # === Lighting (VHAL) ===
    "headlights":        {"type": "vhal", "name": "HEADLIGHTS_STATE",         "area": 0,    "unit": "enum"},
    "high_beam":         {"type": "vhal", "name": "HIGH_BEAM_LIGHTS_STATE",   "area": 0,    "unit": "enum"},
    "hazard_lights":     {"type": "vhal", "name": "HAZARD_LIGHTS_STATE",      "area": 0,    "unit": "enum"},
    "fog_lights_front":  {"type": "vhal", "name": "FRONT_FOG_LIGHTS_STATE",   "area": 0,    "unit": "enum"},
    "fog_lights_rear":   {"type": "vhal", "name": "REAR_FOG_LIGHTS_STATE",    "area": 0,    "unit": "enum"},

    # === Physical state (VHAL) ===
    "door_open_driver":  {"type": "vhal", "name": "DOOR_POS",                 "area": 0x1,  "unit": "bool"},
    "seatbelt_driver":   {"type": "vhal", "name": "SEAT_BELT_BUCKLED",        "area": 0x1,  "unit": "bool"},
    "seat_occupied_driver":{"type":"vhal","name": "SEAT_OCCUPANCY",           "area": 0x1,  "unit": "enum"},
    "fuel_door_open":    {"type": "vhal", "name": "FUEL_DOOR_OPEN",           "area": 0,    "unit": "bool"},
    "outside_temp":      {"type": "vhal", "name": "ENV_OUTSIDE_TEMPERATURE",  "area": 0,    "unit": "°C"},
    "distance_unit":     {"type": "vhal", "name": "DISTANCE_DISPLAY_UNITS",   "area": 0,    "unit": "35=km, 36=miles"},

    # === Environmental sensors (telnet) ===
    "humidity":          {"type": "sensor", "name": "humidity",      "unit": "%"},
    "ambient_temp":      {"type": "sensor", "name": "temperature",   "unit": "°C"},
    "acceleration":      {"type": "sensor", "name": "acceleration",  "unit": "m/s² (x,y,z)"},
    "gyroscope":         {"type": "sensor", "name": "gyroscope",     "unit": "rad/s (x,y,z)"},
    "tilt":              {"type": "sensor", "name": "orientation",   "unit": "deg (x,y,z)"},
}


def _format_key_block() -> str:
    """Flat snake_case key list, 8 per line, for prompts / error messages."""
    keys = sorted(QUERY_REGISTRY.keys())
    return "\n  ".join(", ".join(keys[i:i + 8]) for i in range(0, len(keys), 8))


def _execute_known(q: str) -> str:
    """Execute a query known to be in QUERY_REGISTRY."""
    spec = QUERY_REGISTRY[q]
    try:
        if spec["type"] == "vhal":
            value = get_vhal(spec["name"], spec.get("area", 0))
            if value is None:
                return f"{q}: not available (property may not be supported on this vehicle)"
            return f"{q} = {value} ({spec['unit']})"
        elif spec["type"] == "sensor":
            value = get_sensor(spec["name"])
            if value is None:
                return f"{q}: not available"
            return f"{q} = {value} ({spec['unit']})"
        else:
            return f"Error: unknown query type '{spec['type']}'"
    except Exception as e:
        return f"{q}: query failed ({e})"


def execute_query(query: str) -> str:
    """Execute a single query and return an agent-facing descriptive string.

    Defensively handles three common agent mistakes:
      - comma-separated list of keys -> split and query each
      - "Category (key)" parenthetical form -> extract inner key
      - unknown key -> suggest closest match via difflib
    """
    if not query:
        return (
            "Error: empty query. Pass exactly ONE snake_case name. "
            f"Valid query names:\n  {_format_key_block()}"
        )

    q = query.strip()

    # Defense 1: comma-separated -> query each
    if "," in q:
        parts = [p.strip() for p in q.split(",") if p.strip()]
        lines = [f"  {p}: {execute_query(p)}" for p in parts]
        return (
            "NOTE: Received comma-separated keys. Querying each separately:\n"
            + "\n".join(lines)
            + "\nReminder: Future calls should pass ONE key per call."
        )

    # Defense 2: "Category (key)" -> unwrap the parenthesised inner token
    if "(" in q and ")" in q:
        import re
        m = re.search(r"\(([^)]+)\)", q)
        if m:
            inner = m.group(1).strip()
            return (
                f"NOTE: Interpreted '{q}' as query for '{inner}'. "
                f"Future calls: pass just '{inner}' without parentheses.\n"
                + execute_query(inner)
            )

    # Defense 3: unknown key -> difflib suggest
    q_lower = q.lower()
    if q_lower not in QUERY_REGISTRY:
        import difflib
        suggestions = difflib.get_close_matches(
            q_lower, QUERY_REGISTRY.keys(), n=3, cutoff=0.5)
        hint = f"Did you mean: {', '.join(suggestions)}?\n" if suggestions else ""
        return (
            f"Error: unknown query '{query}'. {hint}"
            f"Valid query names:\n  {_format_key_block()}\n"
            "Pass ONE snake_case name per call. "
            "NO commas, NO category labels, NO parentheses."
        )

    return _execute_known(q_lower)


def list_queries() -> str:
    """Flat snake_case key list for inclusion in agent prompts."""
    return (
        "Available queries (use exactly one of these snake_case names as the "
        "\"query\" parameter):\n  "
        + _format_key_block()
        + "\nPass exactly ONE name per call. NO commas, NO slashes, NO category labels."
    )
