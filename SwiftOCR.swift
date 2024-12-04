import AppKit
import CoreImage
import Foundation
import Vision

let version = "1.0.0"

let options = """
    Options:

        --fast                      Use fast recognition (lower accuracy)
        --languages en,fr,...       Specify recognition languages (ISO 639)
        --correction                Enable language correction
        --custom-words w1,w2,...    Add custom words to improve recognition
        --custom-words-file w.txt   Add custom words from a file (line separated)
    """

let usage = """
    Usage:

        swiftocr <image-path> [options]
        cat <image-path> | swiftocr - [options]

    \(options)

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
    """

struct StandardError: TextOutputStream {
    func write(_ string: String) {
        if let data = string.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

var stderr = StandardError()

struct RecognizedTextResult: Codable {
    let text: String
    let boundingBox: BoundingBox
    let confidence: VNConfidence
}

struct BoundingBox: Codable {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
}

struct OCRConfiguration {
    var imagePath: String = ""
    var useFastRecognition: Bool = false
    var automaticallyDetectsLanguage: Bool = true
    var recognitionLanguages: [String] = ["en"]
    var usesLanguageCorrection: Bool = false
    var customWords: [String] = []
}

func detectText(_ config: OCRConfiguration) {
    let imageData: Data

    if config.imagePath == "-" {
        imageData = FileHandle.standardInput.readDataToEndOfFile()
    } else {
        let imageURL = URL(fileURLWithPath: config.imagePath)
        guard let data = try? Data(contentsOf: imageURL) else {
            print("Failed to load image.", to: &stderr)
            exit(1)
        }
        imageData = data
    }

    guard let nsImage = NSImage(data: imageData),
        let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
    else {
        print("Failed to process image.", to: &stderr)
        exit(1)
    }

    let imageWidth = CGFloat(cgImage.width)
    let imageHeight = CGFloat(cgImage.height)

    var recognizedTextResults: [RecognizedTextResult] = []

    let request = VNRecognizeTextRequest { request, error in
        guard error == nil else {
            print("Error: \(error!.localizedDescription)", to: &stderr)
            exit(1)
        }

        guard let results = request.results as? [VNRecognizedTextObservation], !results.isEmpty
        else {
            print("No text found.", to: &stderr)
            exit(1)
        }

        for observation in results {
            if let topCandidate = observation.topCandidates(1).first {
                let text = topCandidate.string
                let rect = observation.boundingBox
                let confidence = observation.confidence

                let x = Int(rect.origin.x * imageWidth)
                let width = Int(rect.size.width * imageWidth)
                let height = Int(rect.size.height * imageHeight)
                let y = Int(imageHeight - (rect.origin.y * imageHeight) - CGFloat(height))

                let boundingBox = BoundingBox(x: x, y: y, width: width, height: height)

                let result = RecognizedTextResult(
                    text: text, boundingBox: boundingBox, confidence: confidence)

                recognizedTextResults.append(result)
            }
        }
    }

    request.recognitionLevel =
        config.useFastRecognition
        ? VNRequestTextRecognitionLevel.fast : VNRequestTextRecognitionLevel.accurate
    request.automaticallyDetectsLanguage = config.automaticallyDetectsLanguage
    request.recognitionLanguages = config.recognitionLanguages
    request.usesLanguageCorrection = config.usesLanguageCorrection
    request.customWords = config.customWords

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

    do {
        try handler.perform([request])

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        let jsonData = try encoder.encode(recognizedTextResults)
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            print(jsonString)
        }
    } catch {
        print("Failed to perform text detection: \(error.localizedDescription)", to: &stderr)
        exit(1)
    }
}

func parseArguments() throws -> OCRConfiguration {
    let arguments = CommandLine.arguments

    guard arguments.count > 1 else {
        print(usage, to: &stderr)
        exit(1)
    }

    var config = OCRConfiguration()

    var i = 0
    while i < arguments.count {
        let arg = arguments[i]
        switch arg {
        case "--help", "-h":
            print(usage)
            exit(0)
        case "--fast":
            config.useFastRecognition = true
        case "--languages":
            config.automaticallyDetectsLanguage = false

            if i + 1 == arguments.count || arguments[i + 1].hasPrefix("-") {
                print("Missing language list.", to: &stderr)
                exit(1)
            }

            config.recognitionLanguages = arguments[i + 1].split(separator: ",").map(String.init)
            i += 1
        case "--custom-words":
            if i + 1 == arguments.count || arguments[i + 1].hasPrefix("-") {
                print("Missing custom words list.", to: &stderr)
                exit(1)
            }
            config.customWords = arguments[i + 1].split(separator: ",").map(String.init)
            i += 1
        case "--custom-words-file":
            if i + 1 == arguments.count || arguments[i + 1].hasPrefix("-") {
                print("Missing custom words file.", to: &stderr)
                exit(1)
            }
            let filePath = arguments[i + 1]
            do {
                let fileContents = try String(contentsOfFile: filePath, encoding: .utf8)
                config.customWords.append(
                    contentsOf: fileContents.split(separator: "\n").map(String.init))
            } catch {
                print(
                    "Failed to read custom words file: \(error.localizedDescription)", to: &stderr)
                exit(1)
            }
            i += 1
        case "--correction":
            config.usesLanguageCorrection = true
        case "-":
            config.imagePath = "-"
        case "--version":
            print("SwiftOCR v\(version)")
            exit(0)
        default:
            if arg.hasPrefix("-") {
                print("Unknown option: \(arg)\n", to: &stderr)
                print(options, to: &stderr)
                exit(1)
            } else {
                config.imagePath = arg
            }
        }
        i += 1
    }

    guard !config.imagePath.isEmpty else {
        print("Missing image path.", to: &stderr)
        exit(1)
    }

    return config
}

do {
    let config = try parseArguments()
    detectText(config)
} catch {
    print("Error: \(error.localizedDescription)", to: &stderr)
    exit(1)
}
