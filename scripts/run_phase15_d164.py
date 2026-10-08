from consequence_lab.phase15_d164 import run


if __name__ == "__main__":
    result = run()
    print(result["status"])
    print(f"canonical representation gate: {result['canonical_vs_r0']['gate_passed']}")
