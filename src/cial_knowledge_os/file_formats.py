"""Enterprise file-format registry and dataset readiness scanning."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable


class SupportStatus(StrEnum):
    SUPPORTED_NOW = "SUPPORTED_NOW"
    OCR_SUPPORTED = "OCR_SUPPORTED"
    RECOGNIZED_FUTURE_SUPPORT = "RECOGNIZED_FUTURE_SUPPORT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class FileFormatDefinition:
    category_name: str
    category_description: str
    format_label: str
    extensions: tuple[str, ...]
    support_status: SupportStatus
    ingestion_enabled: bool
    requires_ocr: bool = False
    user_facing_message: str = ""
    backend_notes: str = ""
    special_filenames: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["support_status"] = self.support_status.value
        return value


_CATEGORY_DESCRIPTIONS = {
    "Core Enterprise Documents": "Office, PDF, spreadsheet, presentation, CSV, and plain-text business documents.",
    "Technical Documentation": "Structured technical and web-oriented text formats.",
    "Images with OCR": "Raster image files that must pass through OCR before chunking and indexing.",
    "Email / Communication": "Enterprise mail containers and message files reserved for future parsers.",
    "Archives": "Compressed or bundled files reserved for future unpacking and validation.",
    "Source Code": "Application source files reserved for source-aware parsing.",
    "Configuration / DevOps": "Operational configuration files reserved for config-aware parsing.",
    "Multimedia": "Audio and video files reserved for transcription or media analysis.",
    "Engineering / CAD": "Engineering drawings and CAD files reserved for domain-specific extraction.",
}


def _definition(
    category: str,
    label: str,
    extensions: Iterable[str],
    status: SupportStatus,
    *,
    requires_ocr: bool = False,
    special_filenames: Iterable[str] = (),
    notes: str = "",
) -> FileFormatDefinition:
    ingestion_enabled = status in {
        SupportStatus.SUPPORTED_NOW,
        SupportStatus.OCR_SUPPORTED,
    }
    message = {
        SupportStatus.SUPPORTED_NOW: f"{label} is supported for ingestion.",
        SupportStatus.OCR_SUPPORTED: (
            f"{label} is supported through OCR before indexing."
        ),
        SupportStatus.RECOGNIZED_FUTURE_SUPPORT: (
            f"{label} is recognized, but automated ingestion is not yet implemented."
        ),
        SupportStatus.UNSUPPORTED: (
            f"{label} is not currently recognized by CIAL Knowledge OS."
        ),
    }[status]
    normalized_extensions = tuple(
        value.strip().lower().lstrip(".")
        for value in extensions
        if value.strip()
    )
    normalized_special = tuple(
        value.strip().casefold() for value in special_filenames if value.strip()
    )
    return FileFormatDefinition(
        category_name=category,
        category_description=_CATEGORY_DESCRIPTIONS[category],
        format_label=label,
        extensions=normalized_extensions,
        support_status=status,
        ingestion_enabled=ingestion_enabled,
        requires_ocr=requires_ocr,
        user_facing_message=message,
        backend_notes=notes,
        special_filenames=normalized_special,
    )


FORMAT_REGISTRY: tuple[FileFormatDefinition, ...] = (
    _definition("Core Enterprise Documents", "PDF", ("pdf",), SupportStatus.SUPPORTED_NOW),
    _definition("Core Enterprise Documents", "Word", ("docx", "doc"), SupportStatus.SUPPORTED_NOW),
    _definition("Core Enterprise Documents", "Excel", ("xlsx", "xls"), SupportStatus.SUPPORTED_NOW),
    _definition("Core Enterprise Documents", "CSV", ("csv",), SupportStatus.SUPPORTED_NOW),
    _definition("Core Enterprise Documents", "PowerPoint", ("pptx", "ppt"), SupportStatus.SUPPORTED_NOW),
    _definition("Core Enterprise Documents", "Text", ("txt",), SupportStatus.SUPPORTED_NOW),
    _definition("Technical Documentation", "Markdown", ("md", "markdown"), SupportStatus.SUPPORTED_NOW),
    _definition("Technical Documentation", "HTML", ("html", "htm"), SupportStatus.SUPPORTED_NOW),
    _definition("Technical Documentation", "JSON", ("json",), SupportStatus.SUPPORTED_NOW),
    _definition("Technical Documentation", "XML", ("xml",), SupportStatus.SUPPORTED_NOW),
    _definition("Technical Documentation", "YAML", ("yaml", "yml"), SupportStatus.SUPPORTED_NOW),
    _definition("Images with OCR", "PNG", ("png",), SupportStatus.OCR_SUPPORTED, requires_ocr=True),
    _definition("Images with OCR", "JPG/JPEG", ("jpg", "jpeg"), SupportStatus.OCR_SUPPORTED, requires_ocr=True),
    _definition("Images with OCR", "TIFF", ("tiff", "tif"), SupportStatus.OCR_SUPPORTED, requires_ocr=True),
    _definition("Email / Communication", "Email", ("eml", "msg", "pst", "mbox"), SupportStatus.RECOGNIZED_FUTURE_SUPPORT),
    _definition("Archives", "Archives", ("zip", "rar", "7z", "tar", "gz"), SupportStatus.RECOGNIZED_FUTURE_SUPPORT),
    _definition(
        "Source Code",
        "Source Code",
        ("py", "js", "ts", "tsx", "jsx", "java", "cpp", "c", "cs", "go", "rs", "php", "swift", "kt", "rb", "sql"),
        SupportStatus.RECOGNIZED_FUTURE_SUPPORT,
    ),
    _definition(
        "Configuration / DevOps",
        "Configuration Files",
        ("toml", "ini", "env"),
        SupportStatus.RECOGNIZED_FUTURE_SUPPORT,
        special_filenames=("Dockerfile", "docker-compose.yml", "package.json", "requirements.txt"),
    ),
    _definition("Multimedia", "Audio", ("mp3", "wav", "aac"), SupportStatus.RECOGNIZED_FUTURE_SUPPORT),
    _definition("Multimedia", "Video", ("mp4", "mov", "avi", "mkv"), SupportStatus.RECOGNIZED_FUTURE_SUPPORT),
    _definition(
        "Engineering / CAD",
        "CAD / Engineering",
        ("dwg", "dxf", "step", "stp", "iges", "sldprt", "sldasm"),
        SupportStatus.RECOGNIZED_FUTURE_SUPPORT,
    ),
)

_EXTENSION_INDEX = {
    extension: definition
    for definition in FORMAT_REGISTRY
    for extension in definition.extensions
}
_SPECIAL_FILENAME_INDEX = {
    filename: definition
    for definition in FORMAT_REGISTRY
    for filename in definition.special_filenames
}


def get_file_extension(path_or_filename: str | Path) -> str:
    """Return the normalized extension without a dot, or the whole extensionless filename."""

    name = Path(str(path_or_filename)).name.strip()
    suffix = Path(name).suffix
    if suffix:
        return suffix.lstrip(".").casefold()
    if name.startswith(".") and name.count(".") == 1:
        return name.lstrip(".").casefold()
    return name.casefold()


def _lookup_definition(path_or_filename: str | Path) -> FileFormatDefinition | None:
    filename = Path(str(path_or_filename)).name.strip()
    special = _SPECIAL_FILENAME_INDEX.get(filename.casefold())
    if special is not None:
        return special
    return _EXTENSION_INDEX.get(get_file_extension(filename))


def get_file_format_info(path_or_filename: str | Path) -> dict[str, Any]:
    filename = Path(str(path_or_filename)).name
    extension = get_file_extension(filename)
    definition = _lookup_definition(filename)
    if definition is None:
        return {
            "filename": filename,
            "extension": extension,
            "category": "Unsupported",
            "category_name": "Unsupported",
            "category_description": "File types that are not known to the platform.",
            "format_label": "Unsupported",
            "support_status": SupportStatus.UNSUPPORTED.value,
            "ingestion_enabled": False,
            "requires_ocr": False,
            "recognized": False,
            "message": "This file type is not currently recognized by CIAL Knowledge OS.",
            "user_facing_message": "This file type is not currently recognized by CIAL Knowledge OS.",
            "backend_notes": "Add the extension or filename to the central registry before enabling ingestion.",
        }
    value = definition.to_dict()
    value.update(
        {
            "filename": filename,
            "extension": extension,
            "category": definition.category_name,
            "recognized": True,
            "message": definition.user_facing_message,
        }
    )
    return value


def is_supported_file(path_or_filename: str | Path) -> bool:
    return get_file_format_info(path_or_filename)["support_status"] in {
        SupportStatus.SUPPORTED_NOW.value,
        SupportStatus.OCR_SUPPORTED.value,
    }


def is_recognized_file(path_or_filename: str | Path) -> bool:
    return (
        get_file_format_info(path_or_filename)["support_status"]
        != SupportStatus.UNSUPPORTED.value
    )


def validate_ingestion_file(path_or_filename: str | Path) -> dict[str, Any]:
    info = get_file_format_info(path_or_filename)
    status = info["support_status"]
    if status == SupportStatus.SUPPORTED_NOW.value:
        validation = {
            "valid_for_ingestion": True,
            "action": "process",
            "message": "This file can be processed by the ingestion pipeline.",
        }
    elif status == SupportStatus.OCR_SUPPORTED.value:
        validation = {
            "valid_for_ingestion": True,
            "action": "ocr_then_process",
            "message": "This image file can be processed through OCR before indexing.",
        }
    elif status == SupportStatus.RECOGNIZED_FUTURE_SUPPORT.value:
        validation = {
            "valid_for_ingestion": False,
            "action": "skip_with_warning",
            "message": (
                "This file type is recognized by CIAL Knowledge OS, but "
                "automated ingestion is not yet implemented."
            ),
        }
    else:
        validation = {
            "valid_for_ingestion": False,
            "action": "reject",
            "message": "This file type is not currently recognized by CIAL Knowledge OS.",
        }
    return info | validation


def list_supported_formats() -> list[dict[str, Any]]:
    return [
        definition.to_dict()
        for definition in FORMAT_REGISTRY
        if definition.ingestion_enabled
    ]


def list_recognized_formats() -> list[dict[str, Any]]:
    return [definition.to_dict() for definition in FORMAT_REGISTRY]


def list_formats_by_category() -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for definition in FORMAT_REGISTRY:
        grouped.setdefault(
            definition.category_name,
            {
                "category_name": definition.category_name,
                "category_description": definition.category_description,
                "formats": [],
            },
        )["formats"].append(definition.to_dict())
    return dict(sorted(grouped.items()))


def _sample_append(samples: dict[str, list[str]], key: str, filename: str) -> None:
    if len(samples[key]) < 5:
        samples[key].append(filename)


def scan_file_format_readiness(input_dir: str | Path) -> dict[str, Any]:
    root = Path(input_dir).expanduser().resolve()
    paths = sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []
    extension_distribution: Counter[str] = Counter()
    category_distribution: Counter[str] = Counter()
    status_distribution: Counter[str] = Counter()
    sample_files: dict[str, list[str]] = defaultdict(list)
    by_extension: dict[str, dict[str, Any]] = {}
    unsupported_examples: list[dict[str, Any]] = []
    future_examples: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []

    for path in paths:
        relative = path.relative_to(root).as_posix()
        validation = validate_ingestion_file(path.name)
        extension = str(validation["extension"])
        status = str(validation["support_status"])
        category = str(validation["category"])
        extension_distribution[extension] += 1
        category_distribution[category] += 1
        status_distribution[status] += 1
        _sample_append(sample_files, extension, relative)
        row = by_extension.setdefault(
            extension,
            {
                "extension": extension,
                "count": 0,
                "category": category,
                "format_label": validation["format_label"],
                "support_status": status,
                "ingestion_enabled": bool(validation["ingestion_enabled"]),
                "requires_ocr": bool(validation["requires_ocr"]),
                "sample_filenames": [],
            },
        )
        row["count"] += 1
        if len(row["sample_filenames"]) < 5:
            row["sample_filenames"].append(relative)
        if status == SupportStatus.RECOGNIZED_FUTURE_SUPPORT.value:
            item = {
                "filename": path.name,
                "path": relative,
                "extension": extension,
                "category": category,
                "support_status": status,
                "reason": validation["message"],
                "action": validation["action"],
            }
            skipped_files.append(item)
            if len(future_examples) < 10:
                future_examples.append(item)
        elif status == SupportStatus.UNSUPPORTED.value:
            item = {
                "filename": path.name,
                "path": relative,
                "extension": extension,
                "category": category,
                "support_status": status,
                "reason": validation["message"],
                "action": validation["action"],
            }
            skipped_files.append(item)
            if len(unsupported_examples) < 10:
                unsupported_examples.append(item)

    processable = (
        status_distribution[SupportStatus.SUPPORTED_NOW.value]
        + status_distribution[SupportStatus.OCR_SUPPORTED.value]
    )
    return {
        "input_dir": str(root),
        "total_files": len(paths),
        "processable_files": processable,
        "ocr_files": status_distribution[SupportStatus.OCR_SUPPORTED.value],
        "recognized_future_files": status_distribution[
            SupportStatus.RECOGNIZED_FUTURE_SUPPORT.value
        ],
        "unsupported_files": status_distribution[SupportStatus.UNSUPPORTED.value],
        "extension_distribution": dict(sorted(extension_distribution.items())),
        "category_distribution": dict(sorted(category_distribution.items())),
        "support_status_distribution": {
            status.value: status_distribution[status.value]
            for status in SupportStatus
        },
        "unsupported_examples": unsupported_examples,
        "recognized_future_examples": future_examples,
        "sample_files_by_extension": dict(sorted(sample_files.items())),
        "extensions": sorted(
            by_extension.values(),
            key=lambda item: (-int(item["count"]), str(item["extension"])),
        ),
        "skipped_files": skipped_files,
        "registry": list_formats_by_category(),
    }
