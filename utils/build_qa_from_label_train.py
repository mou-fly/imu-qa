import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List


QUESTIONS: List[str] = [
    "Please answer the number of times the user has consumed water during this period.",
    "Please answer whether the user has been drinking water during this period.",
    "Please answer whether the user has taken medication during this period.",
    "Please answer the number of times the user has taken medication during this period.",
    "Please answer whether the user has been reading during this period.",
    "Please answer whether the user read a book before lying down during this period.",
    "Please answer the number of times the user watered the plants during this period.",
    "Please answer whether the user watered the plants during this period.",
    "Please answer whether the user has opened windows for ventilation during this period.",
    "Please answer whether the user was playing with their phone while lying in bed during this period.",
    "Please answer whether the user has been walking during this period.",
    "Please answer the number of times the user has stretched during this period.",
    "Please answer whether the user has eaten fruits during this period.",
    "Please answer the number of times the user has eaten fruits during this period.",
    "Please answer whether the user has wiped the table during this period.",
    "Please answer whether the user has thrown away garbage during this period.",
    "Please answer whether the user washed their hands before eating fruits during this period.",
    "Please answer whether the user washed their hands after eating fruits during this period.",
    "Please answer whether the user has washed their hands after littering during this period.",
    "Please answer whether the user has washed their hands after wiping the table during this period.",
    "Please answer how many times the user washed their hands during this period.",
    "Please answer whether the user operated the mouse during this period.",
    "Please answer whether the user operated the keyboard during this period.",
    "Please answer whether the user has opened the envelope during this period.",
    "Please answer whether the user has turned on the desk lamp during this period.",
    "Please answer whether the user answered the phone during this period.",
    "Please answer the number of times the user answered the phone during this period.",
]


def make_prompt(caption: str) -> str:
    q_lines = [f"({i}) {q}" for i, q in enumerate(QUESTIONS, start=1)]
    return (
        "You are an action-understanding assistant.\n"
        "Given one activity paragraph, answer 27 questions.\n"
        "Rules:\n"
        "- Each answer should be around five words (4-8 words preferred).\n"
        "- Keep each answer short and direct.\n"
        "- If uncertain, still give a brief best guess.\n"
        "- Return STRICT JSON only, keys q1...q27.\n\n"
        f"Paragraph:\n{caption}\n\n"
        "Questions:\n"
        + "\n".join(q_lines)
    )


def parse_answers(raw: str) -> Dict[str, str]:
    text = raw.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Model output is not JSON: {raw[:200]}")
        obj = json.loads(match.group(0))

    result: Dict[str, str] = {}
    for i in range(1, 28):
        key = f"q{i}"
        ans = str(obj.get(key, "")).strip()
        if not ans:
            ans = "insufficient context to confirm"
        result[key] = ans
    return result


def get_client(base_url: str, api_key_env: str, api_key: str = ""):
    key = api_key.strip() if api_key else ""
    if not key:
        key = os.getenv(api_key_env, "")
    if not key:
        raise ValueError(
            f"Missing API key. Please pass --api_key or set environment variable: {api_key_env}"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required. Please run: pip install openai") from exc
    return OpenAI(api_key=key, base_url=base_url)


def ensure_output_header(output_csv: str, fieldnames: List[str]) -> None:
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        return
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_sample_row(writer: csv.DictWriter, caption: str, answers: Dict[str, str]) -> None:
    row = {"text": caption}
    for i in range(1, 28):
        row[f"q{i}"] = answers[f"q{i}"]
    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read label_train.csv and save N x 28 CSV (text + q1..q27)."
    )
    parser.add_argument(
        "--input_csv",
        default=r"C:\\project\\caption\\data\\processed_data_joint_30s_small\\label_train.csv",
        help="Input CSV with columns: id, caption",
    )
    parser.add_argument(
        "--output_csv",
        default=r"C:\\project\\caption\\data\\processed_data_joint_30s_small\\label_train_q27_wide.csv",
        help="Output CSV with 28 columns: text,q1..q27",
    )
    parser.add_argument(
        "--base_url",
        default="https://api.deepseek.com",
        help="OpenAI-compatible base URL (default: DeepSeek)",
    )
    parser.add_argument("--model", default="deepseek-chat", help="Model name (default: cheaper DeepSeek chat model)")
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY", help="API key environment variable")
    parser.add_argument(
        "--api_key",
        default="sk-7f6cc357db9948ffb73a7a3b72ddf20a",
        help="API key string (higher priority than api_key_env)",
    )
    parser.add_argument("--sleep_sec", type=float, default=0.2, help="Sleep between API calls")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional max input rows")
    parser.add_argument("--test", action="store_true", help="Test mode: process first 10 rows")
    args = parser.parse_args()

    if args.test:
        args.max_rows = 10

    client = get_client(args.base_url, args.api_key_env, args.api_key)
    fieldnames = ["text"] + [f"q{i}" for i in range(1, 28)]
    ensure_output_header(args.output_csv, fieldnames)

    processed = 0
    with open(args.input_csv, "r", encoding="utf-8", newline="") as fin, open(
        args.output_csv, "a", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)

        for row in reader:
            if args.max_rows is not None and processed >= args.max_rows:
                break

            caption = str(row.get("caption", "")).strip()
            if not caption:
                continue

            prompt = make_prompt(caption)
            raw = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=420,
            ).choices[0].message.content or "{}"

            answers = parse_answers(raw)
            write_sample_row(writer, caption, answers)
            fout.flush()

            processed += 1
            if processed % 20 == 0:
                print(f"[progress] processed={processed}")
            time.sleep(args.sleep_sec)

    print(f"Done. processed={processed}, output={args.output_csv}")


def test() -> None:
    import sys

    if "--test" not in sys.argv:
        sys.argv.append("--test")
    main()


if __name__ == "__main__":
    main()
