#parser.py
import base64
import csv
import json
import os
import re
import sys
from pathlib import Path

import requests


INPUT_DIR = Path("journal")
OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "books.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:3b"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


PROMPT = """
You are extracting book records from a photograph or scan of a handwritten
reading journal.

Extract ONLY actual book entries.

For every book you can identify, return:
- title
- author

Rules:
1. A book may have an author written after "by".
2. The author may be missing. If so, return an empty string.
3. Ignore check marks, stars, ratings, dates, library names, library
   locations, call numbers, notes, reminders, and other writing that is
   not the book title or author.
4. Do not invent or guess an author.
5. Preserve the title as written as closely as possible.
6. There may be multiple books on one page.
7. Do not treat library information or notes as book titles.
8. If something is clearly a book title but the author cannot be determined,
   include the title with an empty author.

Return ONLY valid JSON in this exact structure:

{
  "books": [
    {
      "title": "Book Title",
      "author": "Author Name"
    }
  ]
}
"""


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_image(path):
    image_data = image_to_base64(path)

    suffix = path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }

    mime_type = mime_types.get(suffix, "image/jpeg")

    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_data],
            }
        ],
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600,
    )

    response.raise_for_status()

    result = response.json()

    content = result["message"]["content"]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse JSON returned for {path}")
        print(content)
        return []

    books = parsed.get("books", [])

    if not isinstance(books, list):
        return []

    return books


def normalize_title(title):
    """
    Creates a comparison key for duplicate detection while preserving
    the original title in the output.
    """
    title = str(title).strip().lower()

    # Normalize curly/smart punctuation.
    title = title.replace("’", "'")
    title = title.replace("“", '"')
    title = title.replace("”", '"')

    # Remove surrounding punctuation.
    title = re.sub(r"^[^\w]+|[^\w]+$", "", title)

    # Collapse whitespace.
    title = re.sub(r"\s+", " ", title)

    return title


def clean_book(book):
    if not isinstance(book, dict):
        return None

    title = str(book.get("title", "")).strip()
    author = str(book.get("author", "")).strip()

    if not title:
        return None

    # Don't allow obvious non-book information through.
    if len(title) > 300:
        return None

    return {
        "title": title,
        "author": author,
    }


def merge_books(books):
    merged = {}

    for book in books:
        book = clean_book(book)

        if book is None:
            continue

        key = normalize_title(book["title"])

        if not key:
            continue

        if key not in merged:
            merged[key] = book
            continue

        existing = merged[key]

        # If the first occurrence has no author but a later occurrence does,
        # keep the later author.
        if not existing["author"] and book["author"]:
            existing["author"] = book["author"]

        # If both have authors and they differ, preserve both rather than
        # silently throwing information away.
        elif (
            existing["author"]
            and book["author"]
            and normalize_title(existing["author"])
            != normalize_title(book["author"])
        ):
            authors = [
                existing["author"],
                book["author"],
            ]

            unique_authors = []

            for author in authors:
                if author not in unique_authors:
                    unique_authors.append(author)

            existing["author"] = "; ".join(unique_authors)

    return list(merged.values())


def get_images():
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    images = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)


def write_csv(books):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Title", "Author"],
        )

        writer.writeheader()

        for book in books:
            writer.writerow(
                {
                    "Title": book["title"],
                    "Author": book["author"],
                }
            )


def main():
    print("Book Journal Parser")
    print("===================")

    images = get_images()

    if not images:
        print(f"No images found in {INPUT_DIR}")
        sys.exit(0)

    print(f"Found {len(images)} journal image(s).")

    all_books = []

    for number, image in enumerate(images, start=1):
        print(
            f"[{number}/{len(images)}] Processing {image}"
        )

        try:
            books = parse_image(image)

            print(
                f"    Found {len(books)} book(s)"
            )

            all_books.extend(books)

        except Exception as e:
            print(
                f"    ERROR processing {image}: {e}"
            )

    print()
    print(f"Raw records: {len(all_books)}")

    books = merge_books(all_books)

    print(f"Unique books: {len(books)}")

    write_csv(books)

    print(f"CSV written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
