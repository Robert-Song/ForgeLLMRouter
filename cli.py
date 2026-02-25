"""
proxy_v6/cli.py
CLI key management menu — extracted from openai_proxy_vLLM_v3.py.
(previous code)
"""

from db import (
    init_db, generate_api_key, add_api_key, remove_api_key,
    update_key_limit, reset_usage, list_api_keys, get_key_limit,
    get_total_usage,
)


def cli_add_key():
    """CLI command to add a new API key."""
    print("\n=== Add New API Key ===")

    choice = input("Generate random key? (y/n): ").strip().lower()

    if choice == 'y':
        api_key = generate_api_key()
        print(f"Generated key: {api_key}")
    else:
        api_key = input("Enter custom API key: ").strip()
        if not api_key:
            print("Error: API key cannot be empty")
            return

    try:
        lifetime_limit = int(input("Enter lifetime token limit: ").strip())
        if lifetime_limit <= 0:
            print("Error: Lifetime limit must be positive")
            return
    except ValueError:
        print("Error: Invalid number")
        return

    if add_api_key(api_key, lifetime_limit):
        print(f"\n✓ API key added successfully!")
        print(f"  Key: {api_key}")
        print(f"  Lifetime Limit: {lifetime_limit:,} tokens")
    else:
        print(f"\n✗ Error: API key already exists")


def cli_remove_key():
    """CLI command to remove an API key."""
    print("\n=== Remove API Key ===")

    api_key = input("Enter API key to remove: ").strip()

    if not api_key:
        print("Error: API key cannot be empty")
        return

    confirm = input(f"Are you sure you want to remove '{api_key}'? (yes/no): ").strip().lower()

    if confirm == 'yes':
        if remove_api_key(api_key):
            print(f"\n✓ API key removed successfully")
        else:
            print(f"\n✗ Error: API key not found")
    else:
        print("Cancelled")


def cli_update_limit():
    """CLI command to update an API key's token limit."""
    print("\n=== Update Token Limit ===")

    api_key = input("Enter API key: ").strip()

    if not api_key:
        print("Error: API key cannot be empty")
        return

    current_limit = get_key_limit(api_key)
    if current_limit is None:
        print(f"✗ Error: API key not found")
        return

    current_usage = get_total_usage(api_key)
    print(f"Current lifetime limit: {current_limit:,} tokens")
    print(f"Current usage: {current_usage:,} tokens")

    try:
        new_limit = int(input("Enter new lifetime token limit: ").strip())
        if new_limit <= 0:
            print("Error: Lifetime limit must be positive")
            return
    except ValueError:
        print("Error: Invalid number")
        return

    if update_key_limit(api_key, new_limit):
        print(f"\n✓ Token limit updated successfully!")
        print(f"  Old limit: {current_limit:,} tokens")
        print(f"  New limit: {new_limit:,} tokens")
        if new_limit < current_usage:
            print(f"  ⚠ Warning: New limit is lower than current usage!")
    else:
        print(f"\n✗ Error: Failed to update limit")


def cli_reset_usage():
    """CLI command to reset usage for an API key."""
    print("\n=== Reset Usage ===")

    api_key = input("Enter API key: ").strip()

    if not api_key:
        print("Error: API key cannot be empty")
        return

    current_usage = get_total_usage(api_key)
    if get_key_limit(api_key) is None:
        print(f"✗ Error: API key not found")
        return

    print(f"Current usage: {current_usage:,} tokens")
    confirm = input(f"Are you sure you want to reset usage to 0? (yes/no): ").strip().lower()

    if confirm == 'yes':
        if reset_usage(api_key):
            print(f"\n✓ Usage reset successfully")
        else:
            print(f"\n✗ Error: Failed to reset usage")
    else:
        print("Cancelled")


def cli_list_keys():
    """CLI command to list all API keys."""
    print("\n=== API Keys ===\n")

    keys = list_api_keys()

    if not keys:
        print("No API keys found")
        return

    for i, key_info in enumerate(keys, 1):
        usage_pct = (key_info['total_usage'] / key_info['lifetime_limit'] * 100) if key_info['lifetime_limit'] > 0 else 0
        print(f"{i}. Key: {key_info['key']}")
        print(f"   Lifetime Limit: {key_info['lifetime_limit']:,} tokens")
        print(f"   Total Usage: {key_info['total_usage']:,} tokens ({usage_pct:.1f}%)")
        print(f"   Remaining: {key_info['remaining']:,} tokens")
        print(f"   Created: {key_info['created_at']}")
        print()


def cli_show_usage():
    """CLI command to show usage for a specific key."""
    print("\n=== Show Key Usage ===")

    api_key = input("Enter API key: ").strip()

    if not api_key:
        print("Error: API key cannot be empty")
        return

    lifetime_limit = get_key_limit(api_key)
    if lifetime_limit is None:
        print(f"✗ Error: API key not found")
        return

    total_usage = get_total_usage(api_key)
    usage_pct = (total_usage / lifetime_limit * 100) if lifetime_limit > 0 else 0

    print(f"\nAPI Key: {api_key}")
    print(f"Lifetime Limit: {lifetime_limit:,} tokens")
    print(f"Total Usage: {total_usage:,} tokens")
    print(f"Remaining: {max(0, lifetime_limit - total_usage):,} tokens")
    print(f"Usage Percentage: {usage_pct:.1f}%")

    if total_usage >= lifetime_limit:
        print("⚠ Status: LIMIT REACHED")
    elif usage_pct >= 90:
        print("⚠ Status: Warning - Near limit")
    else:
        print("✓ Status: Active")


def management_menu():
    """Interactive menu for key management."""
    init_db()

    while True:
        print("\n" + "=" * 50)
        print("API Key Management")
        print("=" * 50)
        print("1. Add new API key")
        print("2. Remove API key")
        print("3. Update token limit")
        print("4. Reset usage for key")
        print("5. List all API keys")
        print("6. Show key usage")
        print("7. Exit")
        print("=" * 50)

        choice = input("\nSelect option (1-7): ").strip()

        if choice == '1':
            cli_add_key()
        elif choice == '2':
            cli_remove_key()
        elif choice == '3':
            cli_update_limit()
        elif choice == '4':
            cli_reset_usage()
        elif choice == '5':
            cli_list_keys()
        elif choice == '6':
            cli_show_usage()
        elif choice == '7':
            print("\nGoodbye!")
            break
        else:
            print("\n✗ Invalid option. Please try again.")
