"""Base exceptions for Certamen Framework.

This module contains the base exception classes used throughout the framework.
Domain-specific exceptions should extend these in the domain layer.
"""


class CertamenError(Exception):
    """Base exception for all Certamen errors."""

    def __init__(self, message: str, *args: object, **kwargs: object) -> None:
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(CertamenError):
    """Exception for configuration errors."""

    pass


class FileSystemError(CertamenError):
    """Exception for file system operations."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.file_path = file_path
        enhanced_message = message

        if file_path:
            enhanced_message = f"[{file_path}] {enhanced_message}"

        super().__init__(enhanced_message, *args)
