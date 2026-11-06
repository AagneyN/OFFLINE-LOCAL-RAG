
import sys

from backend.indexer import index_documents


def main():
    # A document that cannot be read is reported and skipped instead of
    # stopping the whole run.
    report = index_documents(continue_on_error=True)

    if report["failed"]:
        print("\nSome documents could not be indexed:")

        for failure in report["failed"]:
            print(f"  - {failure['document_id']}: {failure['error']}")

        sys.exit(1)


if __name__ == "__main__":
    main()
