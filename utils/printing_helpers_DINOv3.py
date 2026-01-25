def print_report(report):
    ts = report["thresholds"]
    print("\n================ TASK 1 REPORT ================")
    print(f"Config: {report['name']}")
    print(f"Pairs evaluated: {report['pairs_evaluated']}")
    print(f"Valid keypoints: {report['valid_keypoints']}")
    print(f"Layers used (n_layers): {report['n_layers']}")
    print(f"Patch size: {report['patch_size']}")

    print("\nGlobal PCK (per-image mean / macro):")
    for t in ts:
        print(f"  PCK@{t:.2f}: {report['global_pck_macro'][t]*100:.2f}%")

    print("\nGlobal PCK (per-keypoint / micro):")
    for t in ts:
        print(f"  PCK@{t:.2f}: {report['global_pck_micro'][t]*100:.2f}%")

def print_per_category(report):
    ts = report["thresholds"]
    print("\n================ PER-CATEGORY RESULTS ================")
    header = "Category".ljust(15)
    for t in ts:
        header += f" KP@{t:.2f}".rjust(9)
    for t in ts:
        header += f" IMG@{t:.2f}".rjust(10)
    print(header)
    print("-"*len(header))

    for cat, entry in report["per_category"].items():
        row = cat.ljust(15)
        for t in ts:
            row += f"{entry['pck_micro'][t]*100:8.2f}%"
        for t in ts:
            row += f"{entry['pck_macro'][t]*100:9.2f}%"
        print(row)