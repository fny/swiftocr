"""
This is an example Python wrapper for the swiftocr command-line tool.
It happens to be fully featured. The only reason I say it's an example
is that I don't want to make any guarantees about API stability right now.

Use at your own peril. LICENSE.txt applies to this file as well.

```
from swiftocr import SwiftOCR

# Initialize SwiftOCR with the path to the SwiftOCR executable
ocr = SwiftOCR("/path/to/swiftocr")

# Recognize text from an image file or a PIL Image object
file_results = ocr.recognize_file("image.png")
pillow_results = ocr.recognize_pillow(pillow_image)

# Access individual OCR results like a list
result = file_results[0] # => OCRResult
results = file_results[1:3] # => OCRResults
[item.text for item in results]

# Filter by minimum confidence score
file_results.minimum_confidence(0.9) # => OCRResults

# Filter by bounding box coordinates
file_results.within(x=100, y=100, width=200, height=50) # => OCRResults

# Filter by explicit text content
file_results.containing("your query") # => OCRResults

# Search for the closest match to a query string
file_results.search("your query", threshold=0.9, lowercase=True) # => OCRResults
file_results.search_and_score("your query") # => [(score, OCRResult), ...]

# Customize the similarity scoring function
import rapidfuzz
file_results.search("your query", score_func=rapidfuzz.fuzz.ratio) # => OCRResults

# Chain multiple filters together
file_results.minimum_confidence(0.9).within(100, 100, 200, 50).containing("your query")
```
"""

import io
import json
import re
import subprocess
from difflib import SequenceMatcher
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Optional,
    TypedDict,
    Union,
    overload,
)

if TYPE_CHECKING:
    import PIL


class BoundingBoxDict(TypedDict):
    """Represents the structure of a bounding box dictionary."""

    x: int  # X-coordinate of the bounding box
    y: int  # Y-coordinate of the bounding box
    width: int  # Width of the bounding box
    height: int  # Height of the bounding box


class OCRResultDict(TypedDict):
    """Represents the structure of an OCR result dictionary."""

    text: str  # Recognized text
    confidence: float  # Confidence score of the OCR result
    boundingBox: BoundingBoxDict  # Bounding box information for the text


class OCROptions(TypedDict, total=False):
    """Options to configure OCR processing."""

    fast: bool  # Use fast mode for OCR
    languages: list[str]  # List of languages for OCR
    correction: bool  # Enable text correction
    custom_words: list[str]  # List of custom words to include in OCR
    custom_words_file: str  # File containing custom words


class BoundingBox:
    """Represents a bounding box around recognized text."""

    def __init__(self, x: int, y: int, width: int, height: int):
        """Initializes a bounding box with the specified dimensions."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self):
        return f"BoundingBox({self._repr_info})"

    @property
    def center(self) -> tuple[int, int]:
        """Calculates and returns the center coordinates of the bounding box."""
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def top_left(self) -> tuple[int, int]:
        """Returns the top-left corner coordinates of the bounding box."""
        return self.x, self.y

    @property
    def top_right(self) -> tuple[int, int]:
        """Returns the top-right corner coordinates of the bounding box."""
        return self.x + self.width, self.y

    @property
    def bottom_left(self) -> tuple[int, int]:
        """Returns the bottom-left corner coordinates of the bounding box."""
        return self.x, self.y + self.height

    @property
    def bottom_right(self) -> tuple[int, int]:
        """Returns the bottom-right corner coordinates of the bounding box."""
        return self.x + self.width, self.y + self.height

    @property
    def coordinates(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Returns the coordinates of the bounding box corners."""
        return self.top_left, self.top_right, self.bottom_left, self.bottom_right

    @property
    def diagonal(self) -> tuple[int, int, int, int]:
        """Returns the bounding box coordinates for cropping an image."""
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def _repr_info(self) -> str:
        return f"({self.x}, {self.y}), {self.width}x{self.height}"


class OCRResult:
    """Represents the result of OCR processing for a single text block."""

    def __init__(self, text: str, confidence: float, bounding_box: BoundingBox):
        """Initializes an OCR result with text, confidence, and bounding box."""
        self.text = text
        self.confidence = confidence
        self.bounding_box = bounding_box

    @property
    def data(self) -> OCRResultDict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "boundingBox": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
        }

    def __eq__(self, other: Union["OCRResult", str]) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, OCRResult):
            return self.data == other.data
        return False

    def __repr__(self):
        return f"""OCRResult("{self.text}", {self.confidence}, {self.bounding_box._repr_info})"""

    def similarity(self, other: str, lowercase: bool = False) -> float:
        if lowercase:
            return _score_similarity(self.text.lower(), other.lower())
        else:
            return _score_similarity(self.text, other)


def _score_similarity(query: str, target: str):
    return SequenceMatcher(None, query, target).ratio()


class OCRResults:
    """Represents a collection of OCR results."""

    def __init__(self, data: list[OCRResultDict]):
        """Initializes OCR results from a list of OCR result dictionaries."""
        self.data = data
        self.items = [
            OCRResult(
                text=item["text"],
                confidence=item["confidence"],
                bounding_box=BoundingBox(
                    x=item["boundingBox"]["x"],
                    y=item["boundingBox"]["y"],
                    width=item["boundingBox"]["width"],
                    height=item["boundingBox"]["height"],
                ),
            )
            for item in data
        ]

    def __bool__(self) -> bool:
        return bool(self.items)

    @overload
    def __getitem__(self, key: int) -> OCRResult:
        """Handles integer indexing."""
        ...

    @overload
    def __getitem__(self, key: slice) -> "OCRResults":
        """Handles slicing."""
        ...

    def __getitem__(self, key: int | slice) -> Union["OCRResult", "OCRResults"]:
        """Allows access to individual or sliced OCR results."""
        if isinstance(key, int):
            return self.items[key]
        elif isinstance(key, slice):
            return OCRResults(self.data[key])
        else:
            raise TypeError(f"Invalid argument type: {type(key).__name__}")

    def __iter__(self) -> Iterable[OCRResult]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"OCRResults({[item.text for item in self.items]})"

    def __contains__(self, text: str) -> bool:
        """Checks if the OCR results contain a specified text string."""
        return any(text in item.text for item in self.items)

    @property
    def empty(self) -> bool:
        """Checks if the OCR results are empty."""
        return not self.items

    @property
    def exists(self) -> bool:
        """Checks if the OCR results are non-empty."""
        return bool(self.items)

    @property
    def text(self) -> str:
        """Returns the recognized text as a list of strings."""
        return [item.text for item in self.items]

    def minimum_confidence(self, threshold: float) -> "OCRResults":
        """Returns OCR results with a minimum confidence score."""
        return OCRResults(
            [item for item in self.data if item["confidence"] >= threshold]
        )

    def within(self, x: int, y: int, width: int, height: int) -> "OCRResults":
        """Returns OCR results within a specified bounding box."""
        return OCRResults(
            [
                item
                for item in self.data
                if (
                    x <= item["boundingBox"]["x"]
                    and y <= item["boundingBox"]["y"]
                    and x + width
                    >= item["boundingBox"]["x"] + item["boundingBox"]["width"]
                    and y + height
                    >= item["boundingBox"]["y"] + item["boundingBox"]["height"]
                )
            ]
        )

    def containing(self, text: str, lowercase: bool = False) -> "OCRResults":
        """Returns OCR results containing a specified text string."""
        if lowercase:
            return OCRResults(
                [item for item in self.data if text.lower() in item["text"].lower()]
            )
        else:
            return OCRResults([item for item in self.data if text in item["text"]])

    def exactly(self, text: str, lowercase: bool = False) -> "OCRResults":
        """Returns OCR results with an exact text match."""
        if lowercase:
            return OCRResults(
                [item for item in self.data if text.lower() == item["text"].lower()]
            )
        else:
            return OCRResults([item for item in self.data if text == item["text"]])

    def matching(self, pattern: str | re.Pattern, flag: int = 0) -> "OCRResults":
        """Returns OCR results matching a regex pattern."""

        return OCRResults(
            [item for item in self.data if re.match(item["text"], pattern, flag)]
        )

    def filter(self, func) -> "OCRResults":
        """Returns OCR results that satisfy a custom filter function."""
        return OCRResults([item for item in self.data if func(item)])

    def search(
        self,
        query: str,
        threshold: float = 0.0,
        lowercase: bool = False,
        score_func: Callable[[str, str], float] = _score_similarity,
    ) -> "OCRResults":
        """
        Finds the best match for a query string with a given threshold.

        Args:
            query: Query string to search for.
            threshold: Minimum similarity score.
            lowercase: Whether to compare in lowercase.
            score_func: Custom similarity scoring function: f(query, target) -> float.

        Returns:
            OCRResults: Best match for the query string.
        """
        results = self._search_and_score(query, threshold, lowercase, score_func)
        return OCRResults([r[1] for r in results])

    def search_and_score(
        self,
        query: str,
        threshold: float = 0.0,
        lowercase: bool = False,
        score_func: Callable[[str, str], float] = _score_similarity,
    ) -> list[tuple[float, OCRResult]]:
        """
        Finds all matches for a query string while also returning the similarity score.

        Args:
            query: Query string to search for.
            threshold: Minimum similarity score.
            lowercase: Whether to compare in lowercase.
            score_func: Custom similarity scoring function: f(query, target) -> float.

        Returns:
            list[tuple[float, OCRResult]]: List of matches with their similarity scores.
        """
        results = self._search_and_score(query, threshold, lowercase, score_func)
        scores = [r[0] for r in results]
        ocr = OCRResults([r[1] for r in results])
        return list(zip(scores, ocr))

    def first(self) -> Optional[OCRResult]:
        """Returns the first OCR result or None if empty."""
        return self.items[0] if self.items else None

    def last(self) -> Optional[OCRResult]:
        """Returns the last OCR result or None if empty."""
        return self.items[-1] if self.items else None

    def _search_and_score(
        self,
        query: str,
        threshold: float,
        lowercase: bool,
        score_func: Callable[[str, str], float],
    ) -> list[tuple[float, OCRResultDict]]:
        """Search and score that returns raw dictionary data."""
        matches: tuple[float, OCRResultDict] = []
        query = query.lower() if lowercase else query
        query = query.lower() if lowercase else query

        for d in self.data:
            target = d["text"].lower() if lowercase else d["text"]
            score = score_func(query, target)
            if score >= threshold:
                matches.append((score, d))

        return sorted(
            matches,
            key=lambda x: (
                -x[0],
                x[1]["boundingBox"]["x"],
                x[1]["boundingBox"]["y"],
                x[1]["confidence"],
            ),
        )


def _parse_args(options: OCROptions) -> list[str]:
    """Parses OCR options into a list of command-line arguments."""
    args = []
    if options.get("fast"):
        args.append("--fast")

    if "languages" in options:
        languages = ",".join(options["languages"])
        args.extend(["--languages", languages])

    if options.get("correction"):
        args.append("--correction")

    if "custom_words" in options or "custom-words" in options:
        custom_words = ",".join(options["custom_words"])
        args.extend(["--custom-words", custom_words])

    if "custom_words_file" in options or "custom-words-file" in options:
        args.extend(["--custom-words-file", options["custom_words_file"]])

    return args


class SwiftOCR:
    """Wrapper for interacting with the SwiftOCR command-line tool."""

    def __init__(self, swiftocr_path: str):
        """Initializes the SwiftOCR class with the path to the SwiftOCR executable."""
        self.swiftocr_path = swiftocr_path

    def from_file(self, image_path: str, options: OCROptions = {}) -> OCRResults:
        """
        Recognizes text from an image file using SwiftOCR.

        Args:
            image_path: Path to the image file.
            options: Configuration options for OCR (matches SwiftOCR command-line arguments).

        """
        args = [self.swiftocr_path, image_path] + _parse_args(options)

        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )

            result_dict: list[OCRResultDict] = json.loads(result.stdout)

            return OCRResults(result_dict)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"SwiftOCR failed with error:\n{e.stderr.strip()}"
            ) from e
        except json.JSONDecodeError:
            raise ValueError("Failed to parse SwiftOCR output as JSON")

    def from_pillow(
        self, image: "PIL.Image.Image", options: OCROptions = {}
    ) -> OCRResults:
        """Recognizes text from a PIL Image object using SwiftOCR."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        process = subprocess.Popen(
            [self.swiftocr_path, "-"] + _parse_args(options),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(input=buffer.read())

        if process.returncode == 0:
            try:
                result_dict: list[OCRResultDict] = json.loads(stdout)
            except json.JSONDecodeError:
                raise ValueError("Failed to parse SwiftOCR output as JSON")
            return OCRResults(result_dict)
        else:
            raw_message = stderr.decode("utf-8")
            if "No text found" in raw_message:
                return OCRResults([])
            raise RuntimeError(f"SwiftOCR failed: " + stderr.decode("utf-8"))
