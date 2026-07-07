#!/usr/bin/env python3
"""
将 Belle 格式 (instruction/input/output) 转换为 nanochat 对话格式
[{"role":"user","content":...},{"role":"assistant","content":...}]

用法：
    python scripts/convert_belle_to_chatsft.py \
        --input nanochat_data/belle_5k.jsonl \
        --output nanochat_data/chinese_conversations.jsonl \
        --mode merge   # merge=合并到已有文件；replace=覆盖
"""
import argparse
import json
import os
import sys


def belle_to_conversation(item: dict) -> list:
    """把一条 Belle 数据转成 nanochat 对话格式"""
    instruction = item.get("instruction", "").strip()
    inp = item.get("input", "").strip()
    output = item.get("output", "").strip()

    # 拼接 instruction + input 作为用户输入
    user_content = instruction
    if inp:
        user_content = f"{instruction}\n\n{inp}" if instruction else inp

    if not user_content or not output:
        return []

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def is_valid_conversation(conv: list) -> bool:
    """校验对话格式是否合法"""
    if not isinstance(conv, list) or len(conv) < 2:
        return False
    roles = [m.get("role") for m in conv]
    if roles[0] != "user" or roles[1] != "assistant":
        return False
    # 至少有内容
    if not conv[0].get("content") or not conv[1].get("content"):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Belle → nanochat 对话格式转换")
    parser.add_argument("--input", required=True, help="输入 Belle JSONL 文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件")
    parser.add_argument(
        "--mode",
        choices=["merge", "replace"],
        default="merge",
        help="merge=追加到已有输出文件；replace=覆盖",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="过滤掉 user+assistant 文本总长度超过此值的样本（token 长度近似）",
    )
    parser.add_argument(
        "--limit", type=int, default=-1, help="最多转换多少条（-1 表示全部）"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在 {args.input}", file=sys.stderr)
        sys.exit(1)

    # 读取已有数据（merge 模式）
    existing_lines = []
    if args.mode == "merge" and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
        print(f"已有数据 {len(existing_lines)} 条")

    # 转换
    converted = []
    skipped = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if args.limit > 0 and len(converted) >= args.limit:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            conv = belle_to_conversation(item)
            if not is_valid_conversation(conv):
                skipped += 1
                continue

            # 长度过滤
            total_len = sum(len(m["content"]) for m in conv)
            if total_len > args.max_length * 2:  # 中文按 2 字符/token 近似
                skipped += 1
                continue

            converted.append(conv)

    print(f"转换成功 {len(converted)} 条，跳过 {skipped} 条")

    # 写入
    with open(args.output, "w", encoding="utf-8") as f:
        # merge 模式先写已有数据
        if args.mode == "merge":
            for line in existing_lines:
                f.write(line if line.endswith("\n") else line + "\n")
        # 写新数据
        for conv in converted:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    total = len(existing_lines) + len(converted) if args.mode == "merge" else len(converted)
    print(f"输出文件 {args.output} 共 {total} 条对话")


if __name__ == "__main__":
    main()
