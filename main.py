from retrieval import ask
from ingestion import (
    save_uploaded_file,
    add_uploaded_file_to_database
)


def display_menu():
    """Display the main menu."""

    print("\n" + "=" * 60)
    print("          Financial AI Analyzer")
    print("=" * 60)
    print("1. Search Existing Companies")
    print("2. Upload Your Own Financial Data")
    print("3. Exit")
    print("=" * 60)


def upload_data():
    """Upload a financial file and add it to ChromaDB."""

    file_path = input(
        "\nEnter the complete path of your financial file: "
    ).strip()

    if not file_path:
        print("\nNo file path entered.")
        return

    try:

        saved_path = save_uploaded_file(file_path)

        print("\nFile uploaded successfully.")
        print(f"Saved at: {saved_path}")

        print("\nProcessing financial data...")

        document_count, chunk_count = (
            add_uploaded_file_to_database(saved_path)
        )

        print("\nData added successfully!")
        print(f"Documents Created : {document_count}")
        print(f"Chunks Created    : {chunk_count}")

    except Exception as error:
        print(f"\nError: {error}")


def search_company():
    """
    Start the financial question-answer loop.

    Returns:
        "menu" or "exit".
    """

    print("\nSearching existing companies...\n")

    return ask()


def main():

    while True:

        display_menu()

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":

            status = search_company()

            if status == "exit":
                break

        elif choice == "2":

            upload_data()

        elif choice == "3":

            print("\nThank you for using Financial AI Analyzer!")
            break

        else:

            print("\nInvalid choice.")
            print("Please enter 1, 2 or 3.")


if __name__ == "__main__":
    main()