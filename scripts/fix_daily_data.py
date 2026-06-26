"""数据清洗脚本：修复 daily CSV 中的重复行、空日期，并只保留最近2年数据。

用法：
    .venv/bin/python scripts/fix_daily_data.py [--data-root DATA_ROOT] [--years 2] [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


def clean_daily_file(path: Path, cutoff_date: pd.Timestamp, dry_run: bool = False) -> dict:
    """清洗单个 daily CSV 文件。"""
    stats = {"file": path.name, "code": path.stem, "original_rows": 0, "final_rows": 0,
             "null_date_dropped": 0, "duplicates_dropped": 0, "old_data_dropped": 0, "status": "ok"}
    try:
        df = pd.read_csv(path, dtype={"code": str})
    except Exception as e:
        stats["status"] = f"read_error: {e}"
        return stats

    stats["original_rows"] = len(df)

    if df.empty:
        stats["status"] = "empty_file"
        return stats

    # 1. 去除 date 为空的行
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    stats["null_date_dropped"] = before - len(df)

    # 2. 按 (code, date) 去重，保留最后一条
    before = len(df)
    if "code" in df.columns:
        df = df.drop_duplicates(subset=["code", "date"], keep="last")
    else:
        df = df.drop_duplicates(subset=["date"], keep="last")
    stats["duplicates_dropped"] = before - len(df)

    # 3. 只保留 cutoff_date 之后的数据（最近N年）
    before = len(df)
    df = df[df["date"] >= cutoff_date]
    stats["old_data_dropped"] = before - len(df)

    # 4. 排序
    df = df.sort_values("date").reset_index(drop=True)

    # 5. 格式化 date 为字符串（保持 CSV 一致性）
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    stats["final_rows"] = len(df)

    if len(df) == 0:
        stats["status"] = "no_data_after_clean"

    # 写回
    if not dry_run:
        df.to_csv(path, index=False)

    return stats


def main():
    parser = argparse.ArgumentParser(description="修复 daily CSV 数据质量")
    parser.add_argument("--data-root", default="data_real_2000_10pct", help="数据根目录")
    parser.add_argument("--years", type=float, default=2.0, help="保留最近N年数据")
    parser.add_argument("--dry-run", action="store_true", help="只统计不修改文件")
    args = parser.parse_args()

    daily_dir = Path(args.data_root) / "daily"
    if not daily_dir.exists():
        print(f"ERROR: {daily_dir} 不存在")
        return

    # 计算截止日期
    cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=int(args.years * 365)))
    print(f"清洗参数：")
    print(f"  数据目录：{daily_dir}")
    print(f"  保留起始日期：{cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Dry run：{args.dry_run}")
    print()

    files = sorted(daily_dir.glob("*.csv"))
    print(f"发现 {len(files)} 个 CSV 文件")
    print()

    results = []
    total_null = 0
    total_dup = 0
    total_old = 0
    total_empty = 0

    for i, f in enumerate(files, 1):
        stats = clean_daily_file(f, cutoff_date, dry_run=args.dry_run)
        results.append(stats)
        total_null += stats["null_date_dropped"]
        total_dup += stats["duplicates_dropped"]
        total_old += stats["old_data_dropped"]
        if stats["final_rows"] == 0:
            total_empty += 1

        # 每100个输出进度
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] 已处理...")

    print()
    print("=" * 60)
    print("清洗结果汇总")
    print("=" * 60)
    print(f"  文件总数：{len(files)}")
    print(f"  删除空日期行：{total_null:,}")
    print(f"  删除重复行：{total_dup:,}")
    print(f"  删除旧数据行（超过{args.years}年）：{total_old:,}")
    print(f"  清洗后为空的文件：{total_empty}")
    print()

    # 统计清洗后的行数分布
    row_counts = [r["final_rows"] for r in results if r["final_rows"] > 0]
    if row_counts:
        print(f"  有效文件数：{len(row_counts)}")
        print(f"  平均行数：{sum(row_counts) / len(row_counts):.0f}")
        print(f"  最小行数：{min(row_counts)}")
        print(f"  最大行数：{max(row_counts)}")
    print()

    if args.dry_run:
        print("⚠️  DRY RUN 模式，文件未修改。去掉 --dry-run 执行实际清洗。")
    else:
        print("✅ 清洗完成！建议重新运行 quality-check 和 pipeline 验证。")

    # 输出问题文件
    problem_files = [r for r in results if r["status"] != "ok"]
    if problem_files:
        print(f"\n  问题文件（{len(problem_files)} 个）：")
        for r in problem_files[:20]:
            print(f"    {r['code']}: {r['status']}")
        if len(problem_files) > 20:
            print(f"    ... 还有 {len(problem_files) - 20} 个")

    # 保存清洗报告
    report_path = Path(args.data_root) / "quality" / "clean_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(report_path, index=False)
    print(f"\n  清洗报告已保存：{report_path}")


if __name__ == "__main__":
    main()
