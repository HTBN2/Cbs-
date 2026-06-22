"""
Admin CLI — manage client tokens.

Usage:
    python admin.py create  "John"  "Daily 3k"  3000   90000  30
    python admin.py create  "Sara"  "Daily 6k"  6000  180000  30
    python admin.py create  "Ali"   "Daily 10k" 10000 300000  30
    python admin.py list
    python admin.py info    <token>
    python admin.py disable <token>
    python admin.py enable  <token>
    python admin.py delete  <token>
"""

import sys
from db import create_token, list_tokens, disable_token, enable_token, delete_token, get_usage
from datetime import datetime


def cmd_create(args):
    if len(args) < 5:
        print("Usage: python admin.py create <name> <plan> <daily_limit> <total_limit> <days>")
        return
    name, plan, daily, total, days = args[0], args[1], int(args[2]), int(args[3]), int(args[4])
    token = create_token(name, plan, daily, total, days)
    print(f"\nToken created:")
    print(f"  Client : {name}")
    print(f"  Plan   : {plan}")
    print(f"  Daily  : {daily:,}")
    print(f"  Total  : {total:,}")
    print(f"  Days   : {days}")
    print(f"\n  TOKEN  : {token}\n")
    print("  Give this token to your client. They paste it in the extension.\n")


def cmd_list():
    tokens = list_tokens()
    if not tokens:
        print("No tokens.")
        return
    print(f"\n{'CLIENT':<15} {'PLAN':<12} {'DAILY':>8} {'TOTAL':>10} {'EXPIRES':<12} {'STATUS':<8} TOKEN")
    print("-" * 100)
    for t in tokens:
        status = "ACTIVE" if t["is_active"] else "DISABLED"
        expires = t["expires_at"][:10]
        today = datetime.utcnow().date().isoformat()
        if today > expires:
            status = "EXPIRED"
        print(f"{t['client_name']:<15} {t['plan_name']:<12} {t['daily_limit']:>8,} {t['total_limit']:>10,} {expires:<12} {status:<8} {t['token']}")
    print()


def cmd_info(token):
    info = get_usage(token)
    if not info:
        print("Token not found.")
        return
    print(f"\nToken info: {token}")
    print(f"  Client     : {info['client']}")
    print(f"  Plan       : {info['plan']}")
    print(f"  Today used : {info['daily_used']:,} / {info['daily_limit']:,}")
    print(f"  Total used : {info['total_used']:,} / {info['total_limit']:,}")
    print(f"  Expires    : {info['expires_at']}")
    print(f"  Active     : {info['is_active']}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd == "create":
        cmd_create(args)
    elif cmd == "list":
        cmd_list()
    elif cmd == "info" and args:
        cmd_info(args[0])
    elif cmd == "disable" and args:
        disable_token(args[0])
        print(f"Token disabled: {args[0]}")
    elif cmd == "enable" and args:
        enable_token(args[0])
        print(f"Token enabled: {args[0]}")
    elif cmd == "delete" and args:
        delete_token(args[0])
        print(f"Token deleted: {args[0]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
