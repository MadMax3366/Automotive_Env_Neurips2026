"""Smoke test for query_registry."""
from automotive_env.sensors.query_registry import execute_query, QUERY_REGISTRY


def main():
    print(f"Registered {len(QUERY_REGISTRY)} queries")
    print()
    for q in ["speed", "fuel_level", "humidity", "outside_temp",
              "hvac_on", "tire_pressure_fl", "unknown_query_xyz"]:
        result = execute_query(q)
        print(f"  {q:25s} => {result}")


if __name__ == "__main__":
    main()
