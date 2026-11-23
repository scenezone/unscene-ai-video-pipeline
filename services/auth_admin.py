"""Local administrator commands for provisioning and revoking API clients."""
import argparse
import os

from dotenv import load_dotenv

from services.auth_store import AuthStore, DEFAULT_SCOPES


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "rotate", "revoke"))
    parser.add_argument("client_id", nargs="?", help="Existing client ID for rotate/revoke")
    parser.add_argument("--name", help="Client name for create")
    parser.add_argument("--scopes", default=" ".join(DEFAULT_SCOPES),
                        help="Space- or comma-separated scopes for create")
    args = parser.parse_args()
    store = AuthStore(os.getenv("AI_AUTH_DB", "data/auth/auth.sqlite3"))

    if args.action == "create":
        if not args.name:
            parser.error("create requires --name")
        client = store.create_client(args.name, args.scopes)
        print("Client created. Save this secret now; it cannot be retrieved later.")
        print(f"client_id: {client['client_id']}")
        print(f"client_secret: {client['client_secret']}")
        print(f"scopes: {' '.join(client['scopes'])}")
    else:
        if not args.client_id:
            parser.error(f"{args.action} requires client_id")
        try:
            if args.action == "rotate":
                secret = store.rotate_client(args.client_id)
                print("Secret rotated. Previous access tokens were revoked.")
                print(f"client_id: {args.client_id}")
                print(f"client_secret: {secret}")
            else:
                store.revoke_client(args.client_id)
                print(f"Client revoked: {args.client_id}")
        except KeyError:
            parser.error("client not found or already revoked")


if __name__ == "__main__":
    main()
