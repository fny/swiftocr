# SwiftOCR 📖

OCR command line tool for macOS using [Vision Framework](https://developer.apple.com/documentation/vision/).

This works with almost any image format: png, pdf, heic, jpeg, ai, tiff, webp and
more! If you can open it with Preview or
[`NSImage`](https://developer.apple.com/documentation/appkit/nsimage), it should
work.

## Installation

Make sure you have XCode installed. If not, you can install the command line
tools with the following command:

```
xcode-select --install
```

Then simply run `sh bash.sh` to build `swiftocr`, and move it somewhere in your
path like `/usr/local/bin/swiftocr`.

## Usage

```
Usage:

  swiftocr <image-path> [options]
  cat <image-path> | swiftocr - [options]

Options:

    --fast                      Use fast recognition (lower accuracy)
    --languages en,fr,...       Specify recognition languages (ISO 639)
    --correction                Enable language correction
    --custom-words w1,w2,...    Add custom words to improve recognition
    --custom-words-file w.txt   Add custom words from a file (line separated)

Returns the following list with unsorted keys:

    [{
        "text" : str,
        "confidence": float,
        "boundingBox" : {
            "x" : int,
            "y" : int,
            "width" : int,
            "height" : int
        }
    }, ...]

Works on almost any image format.
```

Bounding box values for `x` and `y` start from the top left corner of the image
with `x` increasing to the right and `y` increasing downwards.
